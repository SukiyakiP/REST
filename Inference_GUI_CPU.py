import os
import glob
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import mne
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import savemat
from torch.utils.data import DataLoader

# Import REST dependencies
from RESTCORE import REST
from RESTutils import compute_powers_welch, data_process, create_sequences, viterbi_smooth
import sys

# Determine the base directory
if getattr(sys, 'frozen', False):
    # If running as PyInstaller executable
    BASE_DIR = sys._MEIPASS
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    # If running as a standard python script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPT_DIR = BASE_DIR

# Determine Default Model Path
# 1st Priority: Look in the same folder as the script/executable
local_model = os.path.join(SCRIPT_DIR, "model_general.pth")
if os.path.exists(local_model):
    DEFAULT_MODEL_PATH = local_model
else:
    # 2nd Priority: Fallback to the hardcoded path
    DEFAULT_MODEL_PATH = r"M:\Alex\Python\REST\Model and training data\model_general.pth"
    

FS = 512
EPOCH_LENGTH = 4
WINDOW_SIZE = 90
STEP = 60
BATCH_SIZE = 128
N_CLASSES = 3
F_BIN = 130
DEVICE = torch.device('cpu')

class RESTInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("REST Inference GUI")
        self.root.geometry("1000x800")
        
        # State variables
        self.model = None
        self.model_loaded_path = ""
        self.current_raw_edf = None
        self.current_score = None
        self.current_power = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Top Frame for Model Selection
        model_frame = ttk.LabelFrame(self.root, text="Global Settings", padding=(10, 10))
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="REST Model (.pth):").pack(side=tk.LEFT, padx=5)
        self.model_entry = ttk.Entry(model_frame, width=80)
        self.model_entry.insert(0, DEFAULT_MODEL_PATH)
        self.model_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        self.btn_load_model = ttk.Button(model_frame, text="Load/Update Model", command=self.load_model_thread)
        self.btn_load_model.pack(side=tk.LEFT, padx=5)
        
        self.model_status_label = ttk.Label(model_frame, text="Model Status: Not Loaded", foreground="red")
        self.model_status_label.pack(side=tk.LEFT, padx=10)
        
        # Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.tab_single = ttk.Frame(self.notebook, padding=10)
        self.tab_batch = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.tab_single, text="Single File Mode")
        self.notebook.add(self.tab_batch, text="Batch Mode")
        
        self.setup_single_mode_tab()
        self.setup_batch_mode_tab()
        
        # Bottom Progress Bar
        progress_frame = ttk.Frame(self.root, padding=10)
        progress_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Load default model on startup
        self.load_model_thread()

    # --- Global Functions ---
    def browse_model(self):
        filename = filedialog.askopenfilename(title="Select Model file", filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")])
        if filename:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, filename)
            
    def load_model_thread(self):
        path = self.model_entry.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Model path is invalid.")
            return
            
        self.btn_load_model.config(state=tk.DISABLED)
        self.model_status_label.config(text="Model Status: Loading...", foreground="orange")
        threading.Thread(target=self._load_model, args=(path,), daemon=True).start()
        
    def _load_model(self, path):
        try:
            model = REST(
                in_feat=F_BIN,
                n_classes=N_CLASSES,
                win_len=WINDOW_SIZE,
                d_model=256,
                nhead=8,
                nlayers_epoch=4,
                nlayers_seq=4,
                ff=512,
                fc_hidden1=128,
                fc_hidden2=64,
                dropout=0.1
            ).to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            self.model = model
            self.model_loaded_path = path
            self.root.after(0, self._load_model_success)
        except Exception as e:
            self.root.after(0, self._load_model_fail, str(e))
            
    def _load_model_success(self):
        self.model_status_label.config(text="Model Status: Loaded ✅", foreground="green")
        self.btn_load_model.config(state=tk.NORMAL)
        
    def _load_model_fail(self, error_msg):
        self.model_status_label.config(text="Model Status: Failed ❌", foreground="red")
        self.btn_load_model.config(state=tk.NORMAL)
        messagebox.showerror("Model Load Error", f"Failed to load model:\n{error_msg}")

    # --- Single Mode Tab ---
    def setup_single_mode_tab(self):
        # File selection
        file_frame = ttk.Frame(self.tab_single)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="EDF File:").pack(side=tk.LEFT, padx=5)
        self.single_file_entry = ttk.Entry(file_frame, width=70)
        self.single_file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse", command=self.browse_single_edf).pack(side=tk.LEFT, padx=5)
        
        # Channels
        channel_frame = ttk.LabelFrame(self.tab_single, text="Channels", padding=10)
        channel_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(channel_frame, text="EEG 1:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
        self.cb_eeg1 = ttk.Combobox(channel_frame, state="readonly", width=30)
        self.cb_eeg1.grid(row=0, column=1, padx=5, pady=5)
        
        

        ttk.Label(channel_frame, text="EMG:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.cb_emg = ttk.Combobox(channel_frame, state="readonly", width=30)
        self.cb_emg.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(channel_frame, text="HMM Smoothing:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.E)
        self.hmm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(channel_frame, text="Enable", variable=self.hmm_var).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Actions
        action_frame = ttk.Frame(self.tab_single)
        action_frame.pack(fill=tk.X, pady=10)
        
        self.btn_score_single = ttk.Button(action_frame, text="Score", command=self.run_single_inference_thread, width=15)
        self.btn_score_single.pack(side=tk.LEFT, padx=20)
        
        self.btn_save_single = ttk.Button(action_frame, text="Save Results", command=self.save_single_results, state=tk.DISABLED, width=15)
        self.btn_save_single.pack(side=tk.LEFT, padx=20)
        
        # Plot Frame
        self.plot_frame = ttk.Frame(self.tab_single)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def browse_single_edf(self):
        filename = filedialog.askopenfilename(title="Select EDF file", filetypes=[("EDF files", "*.edf"), ("All files", "*.*")])
        if filename:
            self.single_file_entry.delete(0, tk.END)
            self.single_file_entry.insert(0, filename)
            self.load_edf_channels(filename)
            
    def load_edf_channels(self, filepath):
        self.status_var.set(f"Loading {os.path.basename(filepath)}...")
        self.root.update()
        try:
            self.current_raw_edf = mne.io.read_raw_edf(filepath, preload=False)
            ch_names = self.current_raw_edf.info['ch_names']
            
            self.cb_eeg1['values'] = ch_names
            if ch_names: self.cb_eeg1.current(0)
            
            
            
            self.cb_emg['values'] = ch_names
            if len(ch_names) > 1: self.cb_emg.current(1)
            
            self.status_var.set("EDF Loaded successfully. Please select channels and click Score.")
        except Exception as e:
            messagebox.showerror("EDF Load Error", f"Failed to read EDF channels:\n{str(e)}")
            self.status_var.set("Error loading EDF.")
            
    def run_single_inference_thread(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load the REST model first.")
            return
            
        filepath = self.single_file_entry.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid EDF file.")
            return

        ch_eeg1 = self.cb_eeg1.get()
        
        ch_emg = self.cb_emg.get()
        
        if not ch_eeg1 or not ch_emg:
            messagebox.showerror("Error", "Please select Both EEG1 and EMG channels.")
            return
            
        self.btn_score_single.config(state=tk.DISABLED)
        self.btn_save_single.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Scoring in progress...")
        
        threading.Thread(target=self._run_single_inference, args=(filepath, ch_eeg1, ch_emg, self.hmm_var.get()), daemon=True).start()
        
    def _run_single_inference(self, filepath, ch_eeg1, ch_emg, use_hmm):
        try:
            self.root.after(0, lambda: self.progress_var.set(10))
            raw = mne.io.read_raw_edf(filepath, preload=True)
            
            eeg1_data = raw.get_data(picks=[ch_eeg1])[0]
            emg_data = raw.get_data(picks=[ch_emg])[0]
            
            eeg_data = eeg1_data
                
            self.root.after(0, lambda: self.progress_var.set(30))
            
            # Compute power (Converted to microvolts as in Inference.py)
            power = compute_powers_welch(eeg_data * 1e6, emg_data * 1e6, sfreq=FS)
            
            self.root.after(0, lambda: self.progress_var.set(50))
            
            # STFT processing
            eeg_stft, emg_stft = data_process(eeg_data, emg_data, fs=FS)
            STFT = np.concatenate((eeg_stft, emg_stft), axis=-1)
            
            X = create_sequences(data=STFT, window_size=WINDOW_SIZE, step=STEP)
            sequences_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            sequences_batch = DataLoader(sequences_tensor, batch_size=BATCH_SIZE, shuffle=False)
            
            all_preds = []
            with torch.no_grad():
                for idx, batch_X in enumerate(sequences_batch):
                    batch_X = batch_X.to(DEVICE)
                    output = self.model(batch_X)
                    probs = F.softmax(output, dim=2)
                    first_epoch_probs = probs[:, :STEP, :].cpu().numpy()
                    all_preds.append(first_epoch_probs)
                    
                    # Update progress
                    percent = 50 + int(40 * (idx / len(sequences_batch)))
                    self.root.after(0, lambda p=percent: self.progress_var.set(p))
            
            probs_flat = np.concatenate(all_preds, axis=0).reshape(-1, N_CLASSES)
            
            if use_hmm:
                score = viterbi_smooth(probs_flat) + 1
            else:
                score = np.argmax(probs_flat, axis=1) + 1
                
            self.current_score = score
            self.current_power = power
            
            self.root.after(0, self._single_inference_success, filepath)
            
        except Exception as e:
            self.root.after(0, self._inference_fail, str(e))
            
    def _single_inference_success(self, filepath):
        self.progress_var.set(100)
        self.status_var.set("Scoring completed successfully.")
        self.btn_score_single.config(state=tk.NORMAL)
        self.btn_save_single.config(state=tk.NORMAL)
        
        # Plot Hypnogram
        self.ax.clear()
        time_axis = np.arange(len(self.current_score)) * EPOCH_LENGTH / 3600.0 # Convert to hours
        self.ax.plot(time_axis, self.current_score, drawstyle='steps-mid', color='b')
        self.ax.set_yticks([1, 2, 3])
        self.ax.set_yticklabels(['Wake', 'NREM', 'REM'])
        self.ax.set_xlabel('Time (Hours)')
        self.ax.set_title(f'Hypnogram: {os.path.basename(filepath)}')
        self.ax.set_ylim(0.5, 3.5)
        # Standard hypnogram format inverted per user request
        self.canvas.draw()
        
    def _inference_fail(self, error_msg):
        self.progress_var.set(0)
        self.status_var.set("Scoring Failed.")
        self.btn_score_single.config(state=tk.NORMAL)
        messagebox.showerror("Inference Error", f"An error occurred during scoring:\n{error_msg}")
        
    def save_single_results(self):
        if self.current_score is None: return
        filepath = self.single_file_entry.get()
        folder = os.path.dirname(filepath)
        default_name = os.path.splitext(os.path.basename(filepath))[0] + "_REST_general.mat"
        
        save_path = filedialog.asksaveasfilename(
            initialdir=folder,
            initialfile=default_name,
            title="Save result matrix",
            defaultextension=".mat",
            filetypes=[("MATLAB Files", "*.mat")]
        )
        if save_path:
            savemat(save_path, {'score': self.current_score, 'power': self.current_power})
            self.status_var.set(f"Saved to {os.path.basename(save_path)}")
            messagebox.showinfo("Success", f"Results saved successfully to:\n{save_path}")

    # --- Batch Mode Tab ---
    def setup_batch_mode_tab(self):
        # Folder Selection
        folder_frame = ttk.Frame(self.tab_batch)
        folder_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(folder_frame, text="Target Folder:").pack(side=tk.LEFT, padx=5)
        self.batch_folder_entry = ttk.Entry(folder_frame, width=70)
        self.batch_folder_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(folder_frame, text="Browse", command=self.browse_batch_folder).pack(side=tk.LEFT, padx=5)
        
        # Keywords
        keyword_frame = ttk.LabelFrame(self.tab_batch, text="Channel Matching Keywords (e.g. 'RF' or 'EEG', Case Sensitive)", padding=10)
        keyword_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(keyword_frame, text="EEG 1 Keyword:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
        self.kw_eeg1 = ttk.Entry(keyword_frame, width=20)
        self.kw_eeg1.insert(0, "RF")
        self.kw_eeg1.grid(row=0, column=1, padx=5, pady=5)
        
        

        ttk.Label(keyword_frame, text="EMG Keyword:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.kw_emg = ttk.Entry(keyword_frame, width=20)
        self.kw_emg.insert(0, "EMG")
        self.kw_emg.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(keyword_frame, text="HMM Smoothing:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.E)
        self.batch_hmm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(keyword_frame, text="Enable", variable=self.batch_hmm_var).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Action
        action_frame = ttk.Frame(self.tab_batch)
        action_frame.pack(fill=tk.X, pady=20)
        
        self.btn_batch_score = ttk.Button(action_frame, text="Scan & Score", command=self.run_batch_inference_thread, width=20)
        self.btn_batch_score.pack(side=tk.LEFT, padx=20)
        
        # Log Box
        self.log_box = tk.Text(self.tab_batch, height=15, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def browse_batch_folder(self):
        folder = filedialog.askdirectory(title="Select Target Folder")
        if folder:
            self.batch_folder_entry.delete(0, tk.END)
            self.batch_folder_entry.insert(0, folder)
            
    def _log_batch(self, message):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)
            
    def run_batch_inference_thread(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load the REST model first.")
            return
            
        folder = self.batch_folder_entry.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Please select a valid folder.")
            return

        k_eeg1 = self.kw_eeg1.get().strip()
        
        k_emg = self.kw_emg.get().strip()
        
        if not k_eeg1 or not k_emg:
            messagebox.showerror("Error", "Please provide keywords for Both EEG1 and EMG.")
            return
            
        self.btn_batch_score.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete(1.0, tk.END)
        self.log_box.config(state=tk.DISABLED)
        
        threading.Thread(target=self._run_batch_inference, args=(folder, k_eeg1, k_emg, self.batch_hmm_var.get()), daemon=True).start()
        
    def _run_batch_inference(self, folder, k_eeg1, k_emg, use_hmm):
        edf_files = glob.glob(os.path.join(folder, "**", "*.edf"), recursive=True)
        if not edf_files:
            self.root.after(0, self._log_batch, "No EDF files found in the directory.")
            self.root.after(0, lambda: self.btn_batch_score.config(state=tk.NORMAL))
            return
            
        self.root.after(0, self._log_batch, f"Found {len(edf_files)} EDF files.")
        
        for i, fp_edf in enumerate(edf_files):
            try:
                self.root.after(0, self._log_batch, f"Processing: {os.path.basename(fp_edf)}...")
                self.root.after(0, lambda v=int((i/len(edf_files))*100): self.progress_var.set(v))
                self.root.after(0, lambda f=os.path.basename(fp_edf): self.status_var.set(f"Batch Scoring: {f}"))
                
                raw = mne.io.read_raw_edf(fp_edf, preload=True, verbose=False)
                ch_names = raw.info['ch_names']
                
                eeg1_candidates = [name for name in ch_names if k_eeg1 in name and 'LP' not in name]
                emg_candidates = [name for name in ch_names if k_emg in name]
                
                if not eeg1_candidates:
                    self.root.after(0, self._log_batch, f"  -> Skipped: No channel matching EEG1 keyword '{k_eeg1}'.")
                    continue
                if not emg_candidates:
                    self.root.after(0, self._log_batch, f"  -> Skipped: No channel matching EMG keyword '{k_emg}'.")
                    continue
                    
                ch_eeg1 = eeg1_candidates[0]
                ch_emg = emg_candidates[0]
                
                eeg1_data = raw.get_data(picks=[ch_eeg1])[0]
                emg_data = raw.get_data(picks=[ch_emg])[0]
                
                eeg_data = eeg1_data
                        
                # Score
                power = compute_powers_welch(eeg_data * 1e6, emg_data * 1e6, sfreq=FS)
                eeg_stft, emg_stft = data_process(eeg_data, emg_data, fs=FS)
                STFT = np.concatenate((eeg_stft, emg_stft), axis=-1)
                
                X = create_sequences(data=STFT, window_size=WINDOW_SIZE, step=STEP)
                sequences_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
                sequences_batch = DataLoader(sequences_tensor, batch_size=BATCH_SIZE, shuffle=False)
                
                all_preds = []
                with torch.no_grad():
                    for batch_X in sequences_batch:
                        batch_X = batch_X.to(DEVICE)
                        output = self.model(batch_X)
                        probs = F.softmax(output, dim=2)
                        first_epoch_probs = probs[:, :STEP, :].cpu().numpy()
                        all_preds.append(first_epoch_probs)
                        
                probs_flat = np.concatenate(all_preds, axis=0).reshape(-1, N_CLASSES)
                if use_hmm:
                    score = viterbi_smooth(probs_flat) + 1
                else:
                    score = np.argmax(probs_flat, axis=1) + 1
                    
                # Save
                file_name = os.path.splitext(os.path.basename(fp_edf))[0]
                save_folder = os.path.dirname(fp_edf)
                save_path = os.path.join(save_folder, file_name + "_REST_general.mat")
                
                savemat(save_path, {'score': score, 'power': power})
                self.root.after(0, self._log_batch, f"  -> Success: Saved to {file_name}_REST_general.mat")
                
            except Exception as e:
                self.root.after(0, self._log_batch, f"  -> Error processing {os.path.basename(fp_edf)}: {str(e)}")
                
        self.root.after(0, self._batch_inference_finish)
        
    def _batch_inference_finish(self):
        self.progress_var.set(100)
        self.status_var.set("Batch Scoring Completed.")
        self.btn_batch_score.config(state=tk.NORMAL)
        messagebox.showinfo("Batch Complete", "Batch scoring completed. Check the log for details.")

if __name__ == "__main__":
    root = tk.Tk()
    app = RESTInferenceApp(root)
    root.mainloop()
