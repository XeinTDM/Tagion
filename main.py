# --- START OF FILE main.py ---
# Recommended dependencies:
# pip install numpy opencv-python bchlib-ext tkinter-tooltip
# Note: bchlib-ext is a fork that often provides better wheels for installation.
# If it fails, try 'pip install bchlib'.

import json
import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import bchlib
import cv2
import numpy as np
import tkinter as tk

# --- Constants ---
# Application Info
APP_NAME = "Tagion Watermarking"
APP_VERSION = "3.1 Optimized"

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)

# BCH (Bose-Chaudhuri-Hocquenghem) Code Parameters
# Using a primitive polynomial for GF(2^8) -> x^8 + x^4 + x^3 + x^2 + 1
BCH_POLYNOMIAL = 0x11D
BCH_T_VALUE = 15  # Correctable bits per codeword

# Watermarking Algorithm Parameters
DCT_BLOCK_SIZE = 8
# Mid-frequency DCT coefficients for embedding
DCT_COEFF_1 = (1, 2)
DCT_COEFF_2 = (2, 1)
# Margin to enforce between coefficients for a '1' or '0' bit
EMBED_MARGIN = 20.0

# Detection Search Space Parameters
ROTATION_ANGLES = np.arange(-2.0, 2.01, 0.25)
SCALE_FACTORS = np.linspace(0.90, 1.10, 17)

# --- Configuration ---
DATABASE_FILE = Path("tagion_watermarks.json")
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
JPEG_QUALITY = 90


class BCHCoder:
    """Handles BCH encoding and decoding, encapsulating all related parameters."""

    def __init__(self, t: int, prim_poly: int):
        """
        Initializes the BCH coder.

        Args:
            t (int): The number of correctable errors.
            prim_poly (int): The primitive polynomial for the Galois Field.
        """
        self.bch_instance = None
        self.is_initialized = False
        try:
            m = prim_poly.bit_length() - 1
            self.bch_instance = bchlib.BCH(t=t, prim_poly=prim_poly, m=m)
            self.n = self.bch_instance.n  # Codeword bit length (n)
            self.ecc_bits = self.bch_instance.ecc_bits
            self.k = self.n - self.ecc_bits  # Message bit length (k)
            self.data_bytes = self.k // 8  # Payload capacity in full bytes
            self.max_user_id_len = self.data_bytes
            self.msg_buffer_len = (self.k + 7) // 8  # Expected data buffer size for bchlib
            self.ecc_buffer_len = self.bch_instance.ecc_bytes

            logging.info(
                f"BCH Initialized: n={self.n}, k={self.k} (Payload: {self.data_bytes} B), "
                f"ecc_bits={self.ecc_bits}, t={t}"
            )
            self.is_initialized = True
        except Exception as e:
            logging.error(f"Failed to initialize BCH library: {e}", exc_info=True)
            self.n = 255
            self.ecc_bits = t * 8
            self.k = self.n - self.ecc_bits
            self.data_bytes = self.k // 8
            self.max_user_id_len = self.data_bytes
            self.msg_buffer_len = (self.k + 7) // 8
            self.ecc_buffer_len = (self.ecc_bits + 7) // 8
            logging.warning("BCH operating in fallback mode. Functionality limited.")

    def encode(self, user_id: str) -> np.ndarray | None:
        """Encodes a user ID string into a BCH codeword bit array."""
        if not self.is_initialized:
            return None
        user_id_bytes = user_id.encode('utf-8')
        if len(user_id_bytes) > self.max_user_id_len:
            logging.error(f"User ID too long. Max {self.max_user_id_len} bytes.")
            return None

        padded_msg = user_id_bytes.ljust(self.msg_buffer_len, b"\0")

        try:
            ecc_bytes = self.bch_instance.encode(bytearray(padded_msg))
            msg_bits = np.unpackbits(np.frombuffer(padded_msg, dtype=np.uint8))[:self.k]
            ecc_bits = np.unpackbits(np.frombuffer(ecc_bytes, dtype=np.uint8))[:self.ecc_bits]
            codeword = np.concatenate((msg_bits, ecc_bits))

            if codeword.size != self.n:
                logging.error(f"Codeword generation error: size {codeword.size} != {self.n}")
                return None
            return codeword
        except Exception as e:
            logging.error(f"BCH encoding failed: {e}", exc_info=True)
            return None

    def decode(self, bits: np.ndarray) -> tuple[str | None, int | None]:
        """Decodes a bit array, correcting errors to retrieve the user ID."""
        if not self.is_initialized or bits.size != self.n:
            return None, None

        try:
            received_msg_bits = bits[:self.k]
            received_ecc_bits = bits[self.k:]
            data_packet = np.packbits(received_msg_bits).tobytes()
            ecc_packet = np.packbits(received_ecc_bits).tobytes()
            data_buf = bytearray(data_packet.ljust(self.msg_buffer_len, b'\0'))
            ecc_buf = bytearray(ecc_packet.ljust(self.ecc_buffer_len, b'\0'))
            bit_flips = self.bch_instance.decode(data_buf, ecc_buf)

            if bit_flips == -1:
                return None, None

            payload_bytes = bytes(data_buf)[:self.data_bytes].rstrip(b"\0")
            try:
                decoded_id = payload_bytes.decode("utf-8")
            except UnicodeDecodeError:
                logging.warning(f"Decoded bytes not valid UTF-8: {payload_bytes.hex()}")
                decoded_id = payload_bytes.decode("ascii", errors="ignore")
            
            return decoded_id, bit_flips
        except Exception as e:
            logging.error(f"BCH decoding failed: {e}", exc_info=True)
            return None, None

    @staticmethod
    def bits_to_hex(bit_array: np.ndarray) -> str:
        """Converts a numpy bit array to a hexadecimal string."""
        if bit_array is None:
            return ""
        return np.packbits(bit_array).tobytes().hex()

    def hex_to_bits(self, hex_string: str) -> np.ndarray | None:
        """Converts a hexadecimal string to a numpy bit array of codeword length."""
        try:
            byte_data = bytes.fromhex(hex_string)
            bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
            if bits.size > self.n:
                return bits[:self.n]
            if bits.size < self.n:
                return np.pad(bits, (0, self.n - bits.size), 'constant')
            return bits
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid hex string for conversion: {e}")
            return None


class WatermarkCore:
    """Handles the core image processing for embedding and extracting watermarks."""

    def __init__(self, block_size, coeff1, coeff2, margin):
        self.block_size = block_size
        self.coeff1_pos = coeff1
        self.coeff2_pos = coeff2
        self.margin = margin
        cv2.setUseOptimized(True)

    def embed(self, image_bgr: np.ndarray, watermark_bits: np.ndarray) -> np.ndarray | None:
        """Embeds watermark bits into the luminance channel of an image."""
        if image_bgr is None or image_bgr.size == 0 or watermark_bits is None:
            return None
        try:
            yuv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
            y_channel = yuv_image[:, :, 0]
            
            output_y = y_channel.copy()
            height, width = y_channel.shape
            bit_index, num_bits = 0, watermark_bits.size
            
            for i in range(height // self.block_size):
                for j in range(width // self.block_size):
                    y_start, x_start = i * self.block_size, j * self.block_size
                    block = y_channel[y_start:y_start+self.block_size, x_start:x_start+self.block_size]
                    dct_block = cv2.dct(block)
                    
                    coeff1, coeff2 = dct_block[self.coeff1_pos], dct_block[self.coeff2_pos]
                    bit_to_embed = watermark_bits[bit_index % num_bits]
                    
                    if bit_to_embed == 1:
                        if coeff1 < coeff2 + self.margin:
                            dct_block[self.coeff1_pos] = coeff2 + self.margin
                    else:
                        if coeff1 > coeff2 - self.margin:
                            dct_block[self.coeff1_pos] = coeff2 - self.margin
                            
                    output_y[y_start:y_start+self.block_size, x_start:x_start+self.block_size] = cv2.idct(dct_block)
                    bit_index += 1
            
            yuv_image[:, :, 0] = np.clip(output_y, 0, 255)
            return cv2.cvtColor(yuv_image.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        except Exception as e:
            logging.error(f"Error during embedding: {e}", exc_info=True)
            return None

    def extract(self, image_bgr: np.ndarray, codeword_len: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extracts watermark bits and their confidence scores from an image."""
        if image_bgr is None or image_bgr.size == 0:
            return None, None
        try:
            y_channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
            height, width = y_channel.shape
            votes = np.zeros((codeword_len, 2), dtype=np.float32)
            block_count = 0
            
            for i in range(height // self.block_size):
                for j in range(width // self.block_size):
                    y_start, x_start = i * self.block_size, j * self.block_size
                    block = y_channel[y_start:y_start+self.block_size, x_start:x_start+self.block_size]
                    dct_block = cv2.dct(block)
                    
                    diff = dct_block[self.coeff1_pos] - dct_block[self.coeff2_pos]
                    bit_pos = block_count % codeword_len
                    extracted_bit = 1 if diff > 0 else 0
                    
                    votes[bit_pos, extracted_bit] += abs(diff)
                    block_count += 1
            
            if block_count == 0:
                return None, None

            extracted_bits = (votes[:, 1] > votes[:, 0]).astype(np.uint8)
            total_votes = np.sum(votes, axis=1)
            max_votes = np.max(votes, axis=1)
            confidence = np.zeros_like(total_votes, dtype=float)
            valid_indices = total_votes > 0
            confidence[valid_indices] = max_votes[valid_indices] / total_votes[valid_indices]
            
            return extracted_bits, confidence
        except Exception as e:
            logging.error(f"Error during extraction: {e}", exc_info=True)
            return None, None

    @staticmethod
    def apply_transform(image: np.ndarray, target_size: tuple[int, int], angle: float, scale: float) -> np.ndarray | None:
        """Applies a sequence of geometric transformations to an image."""
        try:
            orig_h, orig_w = image.shape[:2]
            target_w, target_h = target_size
            
            scaled_w, scaled_h = round(orig_w * scale), round(orig_h * scale)
            if scaled_w <= 0 or scaled_h <= 0: return None
            
            inter_scale = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            processed_img = cv2.resize(image, (scaled_w, scaled_h), interpolation=inter_scale)
            
            if angle != 0.0:
                center = (scaled_w / 2, scaled_h / 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                processed_img = cv2.warpAffine(
                    processed_img, rot_mat, (scaled_w, scaled_h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
            
            if (scaled_w, scaled_h) != (target_w, target_h):
                inter_final = cv2.INTER_AREA if (target_w * target_h < scaled_w * scaled_h) else cv2.INTER_CUBIC
                processed_img = cv2.resize(processed_img, (target_w, target_h), interpolation=inter_final)
                
            return processed_img
        except Exception as e:
            logging.error(f"Error during geometric transform: {e}", exc_info=True)
            return None


class DatabaseManager:
    """Manages loading and saving the watermark database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db = self._load()

    def _load(self) -> dict:
        """Loads the JSON database from disk."""
        if not self.db_path.exists():
            return {}
        try:
            with self.db_path.open("r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError("Database is not a dictionary.")
            return data
        except (json.JSONDecodeError, IOError, TypeError) as e:
            logging.error(f"Failed to load or parse database '{self.db_path}': {e}")
            if self.db_path.exists():
                backup_path = self.db_path.with_suffix(".json.corrupt")
                self.db_path.rename(backup_path)
                logging.warning(f"Corrupt database moved to {backup_path}")
            return {}

    def save(self) -> bool:
        """Saves the current database to disk."""
        try:
            with self.db_path.open("w") as f:
                json.dump(self.db, f, indent=2)
            return True
        except (IOError, TypeError) as e:
            logging.error(f"Failed to save database: {e}", exc_info=True)
            return False

    def add_entry(self, hex_key: str, user_id: str, size: tuple[int, int]):
        """Adds or updates an entry in the database."""
        self.db[hex_key] = {'uid': user_id, 'size': list(size)}


class TagionApp(tk.Tk):
    """The main GUI application for watermarking."""
    
    # Optimization: Skip full BCH decoding if confidence is too low.
    # A value of 0.25 is much more effective at filtering noise than 0.05.
    MIN_PASS_RATIO_FOR_DECODE_ATTEMPT = 0.25

    def __init__(self, bch_coder, watermark_core, db_manager):
        super().__init__()

        self.bch_coder = bch_coder
        self.watermark_core = watermark_core
        self.db_manager = db_manager

        self.num_threads = mp.cpu_count()
        self.detection_active = False
        self.detection_stop_event = threading.Event()
        self.interactive_widgets = []

        self._configure_window()
        self._create_widgets()
        
        if not self.bch_coder.is_initialized:
            messagebox.showerror("Fatal Error", "BCH library failed to initialize. Functionality is disabled.")
            self._toggle_controls(False)

    def _configure_window(self):
        self.title(f"{APP_NAME} - {APP_VERSION}")
        self.geometry("620x550")
        self.minsize(600, 500)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TButton', padding=6, relief='flat', background='#ccc')
        style.configure('TNotebook.Tab', padding=(10, 5))

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        embed_frame = self._create_embed_tab(notebook)
        notebook.add(embed_frame, text=" Embed Watermark ")

        detect_frame = self._create_detect_tab(notebook)
        notebook.add(detect_frame, text=" Detect Watermark ")
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=2)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_embed_tab(self, parent):
        frame = ttk.Frame(parent, padding="10")
        
        master_frame = ttk.LabelFrame(frame, text="Input Image", padding="10")
        master_frame.pack(fill=tk.X, pady=5)
        self.master_path_var = tk.StringVar()
        ttk.Label(master_frame, text="Master JPEG:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        w = ttk.Entry(master_frame, textvariable=self.master_path_var, width=45)
        w.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW); self.interactive_widgets.append(w)
        w = ttk.Button(master_frame, text="Browse...", command=self._browse_master)
        w.grid(row=0, column=2, padx=5, pady=5); self.interactive_widgets.append(w)
        master_frame.columnconfigure(1, weight=1)

        uid_frame = ttk.LabelFrame(frame, text="Watermark Data", padding="10")
        uid_frame.pack(fill=tk.X, pady=5)
        self.user_id_var = tk.StringVar()
        max_len = self.bch_coder.max_user_id_len
        ttk.Label(uid_frame, text=f"User ID (max {max_len} B):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        w = ttk.Entry(uid_frame, textvariable=self.user_id_var, width=30)
        w.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW); self.interactive_widgets.append(w)
        uid_frame.columnconfigure(1, weight=1)

        self.embed_button = ttk.Button(frame, text="Embed & Save Watermarked Image", command=self._run_embed)
        self.embed_button.pack(pady=20); self.interactive_widgets.append(self.embed_button)

        return frame

    def _create_detect_tab(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.columnconfigure(0, weight=1)
        
        leak_frame = ttk.LabelFrame(frame, text="Input Image", padding="10")
        leak_frame.pack(fill=tk.X, pady=5)
        self.leak_path_var = tk.StringVar()
        ttk.Label(leak_frame, text="Leaked JPEG:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        w = ttk.Entry(leak_frame, textvariable=self.leak_path_var, width=45)
        w.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW); self.interactive_widgets.append(w)
        w = ttk.Button(leak_frame, text="Browse...", command=self._browse_leak)
        w.grid(row=0, column=2, padx=5, pady=5); self.interactive_widgets.append(w)
        leak_frame.columnconfigure(1, weight=1)
        
        settings_frame = ttk.LabelFrame(frame, text="Detection Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        self.confidence_threshold_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE_THRESHOLD)
        self.conf_label_text = tk.StringVar()
        update_conf_label = lambda v: self.conf_label_text.set(f"Min. Bit Confidence ({float(v):.2f}):")
        update_conf_label(DEFAULT_CONFIDENCE_THRESHOLD)
        ttk.Label(settings_frame, textvariable=self.conf_label_text).grid(row=0, column=0, sticky=tk.W)
        w = ttk.Scale(settings_frame, from_=0.5, to=1.0, orient=tk.HORIZONTAL, variable=self.confidence_threshold_var, length=150, command=update_conf_label)
        w.grid(row=0, column=1, padx=5); self.interactive_widgets.append(w)

        action_frame = ttk.Frame(frame)
        action_frame.pack(pady=10)
        self.detect_button = ttk.Button(action_frame, text="Detect Watermark", command=self._run_detect)
        self.detect_button.grid(row=0, column=0, padx=5); self.interactive_widgets.append(self.detect_button)
        self.stop_button = ttk.Button(action_frame, text="Stop Detection", command=self._stop_detect, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5, ipady=2)
        
        results_frame = ttk.LabelFrame(frame, text="Detection Result", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.result_text = tk.Text(results_frame, height=8, wrap=tk.WORD, bg=self.cget('bg'), relief=tk.FLAT, font=("Courier", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "Awaiting detection...")
        self.result_text.config(state=tk.DISABLED)

        return frame

    def _update_status(self, message: str):
        self.after(0, lambda: self.status_var.set(message))

    def _update_results(self, message: str):
        def task():
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, message)
            self.result_text.config(state=tk.DISABLED)
        self.after(0, task)

    def _toggle_controls(self, enable: bool):
        state = tk.NORMAL if enable else tk.DISABLED
        is_bch_ok = self.bch_coder.is_initialized
        for widget in self.interactive_widgets:
            try:
                if widget in (self.embed_button, self.detect_button):
                    widget.config(state=tk.NORMAL if enable and is_bch_ok else tk.DISABLED)
                else:
                    widget.config(state=state)
            except tk.TclError:
                pass
        
        self.stop_button.config(state=tk.NORMAL if self.detection_active else tk.DISABLED)

    def _browse_master(self):
        path = filedialog.askopenfilename(title="Select Master JPEG", filetypes=[("JPEG", "*.jpg;*.jpeg")])
        if path:
            self.master_path_var.set(path)
            self._update_status(f"Selected master: {Path(path).name}")

    def _browse_leak(self):
        path = filedialog.askopenfilename(title="Select Leaked JPEG", filetypes=[("JPEG", "*.jpg;*.jpeg")])
        if path:
            self.leak_path_var.set(path)
            self._update_status(f"Selected leak candidate: {Path(path).name}")
    
    def _run_embed(self):
        master_path, user_id = self.master_path_var.get(), self.user_id_var.get()
        if not all([master_path, user_id]):
            messagebox.showerror("Input Error", "Please provide a master image and a User ID.")
            return
        if len(user_id.encode('utf-8')) > self.bch_coder.max_user_id_len:
            messagebox.showerror("Input Error", f"User ID is too long (max {self.bch_coder.max_user_id_len} bytes).")
            return

        self._toggle_controls(False)
        self._update_status("Encoding watermark...")
        threading.Thread(target=self._embed_task, args=(master_path, user_id), daemon=True).start()

    def _embed_task(self, master_path, user_id):
        watermark_bits = self.bch_coder.encode(user_id)
        if watermark_bits is None:
            self.after(0, lambda: messagebox.showerror("Error", "Failed to encode User ID."))
            self._update_status("Encoding failed.")
            self.after(0, lambda: self._toggle_controls(True))
            return
            
        try:
            image_bgr = cv2.imread(master_path, cv2.IMREAD_COLOR)
            if image_bgr is None: raise IOError("OpenCV could not read the image.")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("File Error", f"Could not read master image: {e}"))
            self._update_status("Error reading image.")
            self.after(0, lambda: self._toggle_controls(True))
            return

        self._update_status("Embedding watermark...")
        start_time = time.time()
        watermarked_image = self.watermark_core.embed(image_bgr, watermark_bits)
        embed_time = time.time() - start_time
        
        if watermarked_image is None:
            self.after(0, lambda: messagebox.showerror("Error", "Watermark embedding failed. See logs."))
            self._update_status(f"Embedding failed ({embed_time:.2f}s).")
            self.after(0, lambda: self._toggle_controls(True))
            return

        self._update_status(f"Embedded ({embed_time:.2f}s). Prompting for save location...")
        self.after(0, self._prompt_save_embedded, watermarked_image, watermark_bits, user_id, Path(master_path).stem)

    def _prompt_save_embedded(self, image, bits, user_id, stem):
        save_path = filedialog.asksaveasfilename(
            title="Save Watermarked Image",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg;*.jpeg")],
            initialfile=f"{stem}_wm.jpg"
        )
        if not save_path:
            self._update_status("Save cancelled.")
            self._toggle_controls(True)
            return

        try:
            cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            bits_hex = self.bch_coder.bits_to_hex(bits)
            self.db_manager.add_entry(bits_hex, user_id, image.shape[:2][::-1])
            if self.db_manager.save():
                messagebox.showinfo("Success", f"Watermarked image saved to:\n{save_path}")
                self._update_status("Embedding complete.")
            else:
                messagebox.showwarning("DB Warning", "Image saved, but failed to update database.")
                self._update_status("Image saved; DB error.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save image: {e}")
            self._update_status("Error saving image.")
        finally:
            self._toggle_controls(True)

    def _stop_detect(self):
        if self.detection_active:
            self.detection_stop_event.set()
            self._update_status("Stop signal sent. Finishing current tasks...")
            self.stop_button.config(state=tk.DISABLED)

    def _run_detect(self):
        if self.detection_active: return
        leak_path = self.leak_path_var.get()
        if not leak_path:
            messagebox.showerror("Input Error", "Please select an image file to check.")
            return

        self.db_manager.db = self.db_manager._load()
        if not self.db_manager.db:
            messagebox.showinfo("Info", "Watermark database is empty or not found. Cannot detect.")
            return

        self.detection_active = True
        self.detection_stop_event.clear()
        self._toggle_controls(False)
        self._update_results("Starting detection...")
        threading.Thread(target=self._detect_task, args=(leak_path,), daemon=True).start()

    def _detect_task(self, leak_path: str):
        start_time = time.time()
        try:
            leaked_image_bgr = cv2.imread(leak_path, cv2.IMREAD_COLOR)
            if leaked_image_bgr is None: raise IOError("OpenCV could not read the image.")
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("File Error", f"Could not read image: {e}"))
            self._finalize_detection(None, time.time() - start_time, 0, 0, error=True)
            return

        db_sizes = {tuple(v['size']) for v in self.db_manager.db.values() if isinstance(v, dict) and 'size' in v}
        h, w = leaked_image_bgr.shape[:2]
        if w > self.watermark_core.block_size and h > self.watermark_core.block_size:
            db_sizes.add((w, h))

        if not db_sizes:
            self._update_results("Error: No valid target sizes in database or from image.")
            self._finalize_detection(None, time.time() - start_time, 0, 0, error=True)
            return
            
        priority_tasks, remaining_tasks = [], []
        checked_params = set()

        # Prioritize likely transformations (no/minimal change)
        for size in db_sizes:
            for angle in [0.0, -0.25, 0.25]:
                for scale in [1.0, 0.98, 1.02]:
                    params = (size, angle, scale)
                    if params not in checked_params:
                        priority_tasks.append((leaked_image_bgr, *params))
                        checked_params.add(params)
        
        # Create the list for the exhaustive search
        for size in db_sizes:
            for angle in ROTATION_ANGLES:
                for scale in SCALE_FACTORS:
                    params = (size, angle, scale)
                    if params not in checked_params:
                        remaining_tasks.append((leaked_image_bgr, *params))
                        checked_params.add(params)
        
        total_task_count = len(priority_tasks) + len(remaining_tasks)
        self.after(0, lambda: self.progress_bar.config(maximum=total_task_count))
        self._update_status(f"Checking {total_task_count} transforms on {self.num_threads} threads...")
        
        best_match = None
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            task_batches = [priority_tasks, remaining_tasks]
            
            for batch in task_batches:
                if not batch or self.detection_stop_event.is_set():
                    break
                
                futures = {executor.submit(self._process_geometry_task, *task): task for task in batch}
                
                for future in as_completed(futures):
                    if self.detection_stop_event.is_set() and (best_match is None or best_match[1] > 0):
                        pass # Allow running tasks to finish
                    elif self.detection_stop_event.is_set():
                        break

                    try:
                        result = future.result()
                        if result:
                            _uid, flips, _conf, _angle, _scale, _size = result
                            if best_match is None or flips < best_match[1]:
                                best_match = result
                            if flips == 0:
                                self._update_status("Perfect match found! Stopping search.")
                                self.detection_stop_event.set()
                    except Exception as exc:
                        logging.error(f"Detection sub-task failed: {exc}", exc_info=True)
                    
                    processed_count += 1
                    self.after(0, lambda p=processed_count: self.progress_var.set(p))
            
            # Cancel any remaining futures if stopped early
            for f in futures:
                if not f.done():
                    f.cancel()

        self._finalize_detection(best_match, time.time() - start_time, processed_count, total_task_count)

    def _process_geometry_task(self, image, size, angle, scale):
        """Worker function for a single geometry transform and extraction attempt."""
        if self.detection_stop_event.is_set(): return None

        transformed_img = self.watermark_core.apply_transform(image, size, angle, scale)
        if transformed_img is None: return None
        
        bits, confidence = self.watermark_core.extract(transformed_img, self.bch_coder.n)
        if bits is None: return None
        
        conf_threshold = self.confidence_threshold_var.get()
        pass_ratio = np.mean(confidence >= conf_threshold)
        if pass_ratio < self.MIN_PASS_RATIO_FOR_DECODE_ATTEMPT:
            return None
            
        user_id, flips = self.bch_coder.decode(bits)
        if user_id is None or flips is None:
            return None

        re_encoded_bits = self.bch_coder.encode(user_id)
        if re_encoded_bits is not None:
            re_encoded_hex = self.bch_coder.bits_to_hex(re_encoded_bits)
            if re_encoded_hex in self.db_manager.db:
                return user_id, flips, pass_ratio, angle, scale, size
        return None

    def _finalize_detection(self, match, duration, processed, total, error=False):
        """Updates the GUI after the detection task is complete."""
        if error:
            status = "Detection failed."
        elif match:
            user_id, flips, conf, angle, scale, size = match
            robustness = "Perfect (0 bitflips)" if flips == 0 else f"Corrected ({flips} bitflips)"
            result_msg = (
                f"** MATCH FOUND **\n\n"
                f"  User ID:         {user_id}\n"
                f"  Correction:      {robustness}\n"
                f"  Confidence Pass: {conf*100:.1f}% (bits >= {self.confidence_threshold_var.get():.2f})\n\n"
                f"  Detected Transform:\n"
                f"    Target Size:   {size[0]}x{size[1]} px\n"
                f"    Rotation:      {angle:+.2f}°\n"
                f"    Scale Factor:  {scale:.3f}"
            )
            self._update_results(result_msg)
            status = f"Match found! ({duration:.2f}s)"
        elif self.detection_stop_event.is_set() and processed < total:
            self._update_results("Detection stopped by user.")
            status = f"Detection stopped ({duration:.2f}s)"
        else:
            self._update_results("No matching watermark found after exhaustive search.")
            status = f"No match found ({duration:.2f}s)"

        self._update_status(status)
        self.detection_active = False
        self.after(0, lambda: self._toggle_controls(True))
        self.after(0, lambda: self.progress_var.set(0))

    def _on_closing(self):
        """Handle window close event."""
        if self.detection_active:
            if messagebox.askokcancel("Quit", "Detection is in progress. Are you sure you want to quit?"):
                self.detection_stop_event.set()
                self.destroy()
        else:
            self.destroy()


if __name__ == "__main__":
    try:
        bch_coder = BCHCoder(t=BCH_T_VALUE, prim_poly=BCH_POLYNOMIAL)
        watermark_core = WatermarkCore(
            block_size=DCT_BLOCK_SIZE,
            coeff1=DCT_COEFF_1,
            coeff2=DCT_COEFF_2,
            margin=EMBED_MARGIN
        )
        db_manager = DatabaseManager(DATABASE_FILE)
        
        app = TagionApp(bch_coder, watermark_core, db_manager)
        app.mainloop()
    except Exception as e:
        logging.critical(f"Fatal error starting application: {e}", exc_info=True)
        try:
            root_err = tk.Tk()
            root_err.withdraw()
            messagebox.showerror("Application Startup Error", f"A critical error occurred:\n{e}")
        except Exception:
            pass