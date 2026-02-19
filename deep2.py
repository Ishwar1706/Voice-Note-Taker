# --- Required Imports ---
import os
import whisper
import numpy as np
import sounddevice as sd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from fpdf import FPDF
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from sklearn.cluster import KMeans
import requests
import json




# --- Use local english.pickle for sentence tokenization ---
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
punkt_paths = [
    os.path.join(nltk_data_path, 'tokenizers', 'punkt', 'english.pickle'),
    os.path.join(nltk_data_path, 'tokenizers', 'punkt', 'PY3', 'english.pickle'),
]
for path in punkt_paths:
    if os.path.exists(path):
        punkt_path = path
        break
else:
    raise RuntimeError(
        "Could not find english.pickle for sentence tokenization.\n"
        "Please place it in one of these locations:\n"
        + '\n'.join(punkt_paths)
    )
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pickle
with open(punkt_path, 'rb') as f:
    punkt_tokenizer = pickle.load(f)
def local_sent_tokenize(text):
    return punkt_tokenizer.tokenize(text)

# ================== CONFIGURATION ==================
OPENROUTER_API_KEY = "Your api key"  # ⚠️ Replace with your actual OpenRouter API key
DEEPSEEK_MODEL = "deepseek-chat"          # Free model available via OpenRouter
# ===================================================

class LectureProcessor:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.kw_model = KeyBERT()
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lecture-notes-app.com",
            "X-Title": "AI Lecture Note Taker"
        }

    def process_audio(self, audio_path, language="en"):
        result = self.model.transcribe(audio_path, language=language)
        text = result["text"]
        keywords = self.kw_model.extract_keywords(text)
        return {"text": text, "keywords": keywords}

    def summarize_with_llm(self, text):
        try:
            prompt = f"""Create a concise bullet-point summary of this lecture:
            
            {text}
            
            Focus on key concepts, main arguments, and important details. 
            Use 5-7 bullet points maximum."""
            
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=20
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return None
        except Exception:
            return None

    def cluster_topics_with_llm(self, text):
        try:
            prompt = f"""Analyze this lecture transcript and identify 3-5 main topics:
            
            {text}
            
            For each topic:
            1. Give a short title (2-5 words)
            2. List 2-3 key points
            3. Include brief examples
            
            Format with clear headings and bullet points."""
            
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 800
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=20
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return None
        except Exception:
            return None

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.sample_rate = 16000

    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.callback
        )
        self.stream.start()

    def callback(self, indata, frames, time, status):
        if self.is_recording:
            self.frames.append(indata.copy())

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.stream.stop()
            audio_data = np.concatenate(self.frames)
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(filename, audio_data, self.sample_rate)
            return filename
        return None

class LectureNoteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.processor = LectureProcessor()
        self.recorder = AudioRecorder()
        self.sessions = []
        self.highlighted_keywords = []
        self.elapsed_seconds = 0
        self.update_timer = None

        self.title(f"AI Lecture Note Taker (Powered by {DEEPSEEK_MODEL})")
        self.geometry("1100x720")
        self.configure(bg="#fefeff")
        self.create_widgets()

    def create_widgets(self):
        # Header
        header = tk.Frame(self, bg="#5c4d9d")
        header.pack(fill=tk.X)
        tk.Label(header, text="Lecture Note Taker", font=("Segoe UI", 20, "bold"), 
                 bg="#5c4d9d", fg="white").pack(pady=20)

        # Control Panel
        control_frame = tk.Frame(self, bg="#ecebfd")
        control_frame.pack(fill=tk.X, pady=5)

        # Language Selection
        self.language_var = tk.StringVar(value="en")
        tk.Label(control_frame, text="Language:", bg="#ecebfd").grid(row=0, column=0, padx=5)
        ttk.Combobox(control_frame, textvariable=self.language_var, 
                    values=["en", "hi", "mr"], width=6).grid(row=0, column=1, padx=5)

        # Status Indicators
        self.timer_label = tk.Label(control_frame, text="Duration: 00:00", bg="#ecebfd")
        self.timer_label.grid(row=0, column=2, padx=10)
        self.status_label = tk.Label(control_frame, text="Status: Ready", bg="#ecebfd")
        self.status_label.grid(row=0, column=3, padx=10)

        # Buttons
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_recording)
        self.start_btn.grid(row=0, column=4, padx=5)
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=5, padx=5)
        self.import_btn = ttk.Button(control_frame, text="Import", command=self.import_audio)
        self.import_btn.grid(row=0, column=6, padx=5)
        self.export_btn = ttk.Button(control_frame, text="Export", command=self.export_transcript)
        self.export_btn.grid(row=0, column=7, padx=5)
        self.summarize_btn = ttk.Button(control_frame, text="Summarize", command=self.summarize_transcript)
        self.summarize_btn.grid(row=0, column=8, padx=5)
        self.topics_btn = ttk.Button(control_frame, text="Topics", command=self.cluster_topics)
        self.topics_btn.grid(row=0, column=9, padx=5)

        # Main Content
        content = tk.Frame(self, bg="#fefeff")
        content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Session History
        left_panel = tk.Frame(content, bg="#f8f8ff", width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(left_panel, text="Session History", bg="#dcdcff", 
                 font=("Segoe UI", 11, "bold")).pack(fill=tk.X)
        self.session_listbox = tk.Listbox(left_panel)
        self.session_listbox.pack(fill=tk.BOTH, expand=True)
        self.session_listbox.bind("<<ListboxSelect>>", self.load_selected_session)

        # Transcript Display
        main_panel = tk.Frame(content, bg="#ffffff")
        main_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.transcript_box = tk.Text(main_panel, wrap=tk.WORD, font=("Segoe UI", 11))
        self.transcript_box.pack(fill=tk.BOTH, expand=True)
        self.transcript_box.tag_config("highlight", background="yellow", foreground="black")

    def update_duration(self):
        mins, secs = divmod(self.elapsed_seconds, 60)
        self.timer_label.config(text=f"Duration: {mins:02d}:{secs:02d}")
        self.elapsed_seconds += 1
        self.update_timer = self.after(1000, self.update_duration)

    def start_recording(self):
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Recording...")
        self.elapsed_seconds = 0
        self.update_duration()
        threading.Thread(target=self.recorder.start_recording, daemon=True).start()

    def stop_recording(self):
        filename = self.recorder.stop_recording()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.update_timer:
            self.after_cancel(self.update_timer)
        self.status_label.config(text="Status: Processing...")
        if filename:
            threading.Thread(target=self.process_audio, args=(filename,), daemon=True).start()

    def import_audio(self):
        filetypes = [("Audio Files", "*.wav *.mp3"), ("All Files", "*.*")]
        filename = filedialog.askopenfilename(title="Select Audio File", filetypes=filetypes)
        if filename:
            self.status_label.config(text="Status: Importing audio...")
            threading.Thread(target=self.process_audio, args=(filename,), daemon=True).start()

    def process_audio(self, audio_path):
        try:
            result = self.processor.process_audio(audio_path, language=self.language_var.get())
            self.transcript_box.delete(1.0, tk.END)
            self.highlighted_keywords = [kw[0] for kw in result["keywords"]]
            self.transcript_box.insert(tk.END, result["text"])
            self.highlight_keywords()
            self.save_session(audio_path, result["text"])
            self.status_label.config(text="Status: Done")
        except Exception as e:
            self.status_label.config(text="Status: Error")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

    def highlight_keywords(self):
        text = self.transcript_box.get(1.0, tk.END).lower()
        for kw in self.highlighted_keywords:
            start = 1.0
            while True:
                pos = self.transcript_box.search(kw.lower(), start, tk.END)
                if not pos:
                    break
                end = f"{pos}+{len(kw)}c"
                self.transcript_box.tag_add("highlight", pos, end)
                start = end

    def export_transcript(self):
        content = self.transcript_box.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("No Content", "Transcript is empty!")
            return

        options = [("TXT file", "*.txt"), ("PDF file", "*.pdf")]
        path = filedialog.asksaveasfilename(title="Export Transcript", defaultextension=".txt", filetypes=options)
        if not path:
            return
        if path.endswith(".pdf"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in content.split("\n"):
                pdf.multi_cell(0, 10, line)
            pdf.output(path)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        messagebox.showinfo("Exported", f"Transcript exported successfully to:\n{path}")

    def summarize_transcript(self):
        content = self.transcript_box.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Empty", "No transcript to summarize")
            return

        self.status_label.config(text="Status: Summarizing...")

        def do_summarize():
            try:
                llm_summary = self.processor.summarize_with_llm(content)
                if llm_summary and llm_summary.strip():
                    summary = llm_summary.strip()
                else:
                    import re
                    sentences = local_sent_tokenize(content)
                    keywords = [kw.lower() for kw in self.highlighted_keywords]
                    # Remove duplicates and very short sentences
                    filtered = [s for s in sentences if len(s.strip()) > 25]
                    seen = set()
                    unique = []
                    for s in filtered:
                        s_norm = re.sub(r'\W+', '', s.strip().lower())
                        if s_norm not in seen:
                            unique.append(s)
                            seen.add(s_norm)
                    # Score sentences by keyword density
                    scored = []
                    for s in unique:
                        kw_count = sum(1 for kw in keywords if kw in s.lower())
                        scored.append((kw_count, s))
                    # Prefer sentences with multiple keywords
                    top = [s for count, s in sorted(scored, reverse=True) if count > 1]
                    # If not enough, add single-keyword sentences
                    if len(top) < 5:
                        top += [s for count, s in sorted(scored, reverse=True) if count == 1 and s not in top]
                    # If still not enough, use most central sentences
                    if len(top) < 5:
                        def sentence_similarity(a, b):
                            a_set = set(a.lower().split())
                            b_set = set(b.lower().split())
                            return len(a_set & b_set) / (1 + len(a_set | b_set))
                        centrality_scores = []
                        for i, s in enumerate(unique):
                            centrality = sum(sentence_similarity(s, other) for j, other in enumerate(unique) if i != j)
                            centrality_scores.append((centrality, s))
                        for _, s in sorted(centrality_scores, reverse=True):
                            if s not in top:
                                top.append(s)
                            if len(top) >= 7:
                                break
                    # Limit redundancy: only one sentence per topic/section
                    final = []
                    topic_seen = set()
                    for s in top:
                        topic_kw = None
                        for kw in keywords:
                            if kw in s.lower():
                                topic_kw = kw
                                break
                        if topic_kw:
                            if topic_kw in topic_seen:
                                continue
                            topic_seen.add(topic_kw)
                        final.append(s)
                        if len(final) >= 7:
                            break
                    if not final:
                        final = unique[:7]
                    if not final:
                        summary = "No key points found."
                    else:
                        summary = '\n- ' + '\n- '.join(final)
                self.after(0, lambda: self.show_result("Summary", summary))
                self.status_label.config(text="Status: Ready")
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.status_label.config(text="Status: Error")

        threading.Thread(target=do_summarize, daemon=True).start()

    def cluster_topics(self):
        content = self.transcript_box.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Empty", "No transcript to analyze")
            return

        self.status_label.config(text="Status: Analyzing topics...")

        def do_cluster():
            try:
                llm_topics = self.processor.cluster_topics_with_llm(content)
                if llm_topics and llm_topics.strip():
                    topics = llm_topics.strip()
                else:
                    sentences = local_sent_tokenize(content)
                    if len(sentences) < 3:
                        topics = "Not enough content for topic analysis."
                    else:
                        try:
                            model = SentenceTransformer('all-MiniLM-L6-v2')
                            n_clusters = min(5, max(2, len(sentences)//7))
                            embeddings = model.encode(sentences)
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            clusters = kmeans.fit_predict(embeddings)
                            topic_texts = []
                            for i in range(n_clusters):
                                topic_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]
                                if topic_sentences:
                                    kw_model = KeyBERT()
                                    topic_keywords = kw_model.extract_keywords(' '.join(topic_sentences), top_n=1)
                                    topic_title = topic_keywords[0][0] if topic_keywords else f"Topic {i+1}"
                                    topic_preview = "\n".join(f"- {s}" for s in topic_sentences[:5])
                                    topic_texts.append(f"{topic_title}:\n{topic_preview}")
                            if topic_texts:
                                topics = "\n\n".join(topic_texts)
                            else:
                                topics = "No topics could be clustered from the transcript."
                        except Exception as e:
                            topics = f"Topic clustering failed: {str(e)}"
                self.after(0, lambda: self.show_result("Topic Analysis", topics))
                self.status_label.config(text="Status: Ready")
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.status_label.config(text="Status: Error")

        threading.Thread(target=do_cluster, daemon=True).start()

    def show_result(self, title, content):
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("700x500")
        
        text_frame = tk.Frame(window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)
        
        text.insert(tk.END, content)
        text.config(state=tk.DISABLED)
        
        btn_frame = tk.Frame(window)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Copy", 
                   command=lambda: [self.clipboard_clear(), 
                                    self.clipboard_append(content)]).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", 
                   command=window.destroy).pack(side=tk.RIGHT, padx=5)

    def save_session(self, audio_path, transcript):
        session_name = os.path.basename(audio_path)
        self.sessions.append((session_name, transcript))
        self.session_listbox.insert(tk.END, session_name)

    def load_selected_session(self, event):
        if not self.session_listbox.curselection():
            return
        index = self.session_listbox.curselection()[0]
        _, transcript = self.sessions[index]
        self.transcript_box.delete(1.0, tk.END)
        self.transcript_box.insert(tk.END, transcript)
        self.highlight_keywords()

if __name__ == "__main__":
    app = LectureNoteApp()
    app.mainloop()