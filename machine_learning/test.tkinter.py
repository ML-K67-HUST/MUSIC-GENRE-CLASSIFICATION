import tkinter as tk
from tkinter import filedialog, messagebox
import os
import warnings
from predict import predict_

# Suppress warnings
warnings.filterwarnings('ignore')

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Music Genre Prediction")
        self.geometry("700x500")  # Larger window size

        self.upload_button = tk.Button(self, text="Upload Song", command=self.upload_file, font=("Helvetica", 14))
        self.upload_button.pack(pady=40)  # Increased padding

        # Larger predict button
        self.predict_button = tk.Button(self, text="Predict Genre", command=self.predict, state=tk.DISABLED, font=("Helvetica", 16))
        self.predict_button.pack(pady=30)

        # Larger text widget
        self.result_text = tk.Text(self, wrap=tk.WORD, height=15, width=80, font=("Helvetica", 12))  # Larger font
        self.result_text.pack(pady=20)
        self.result_text.config(state=tk.DISABLED)

        self.uploaded_file_path = None

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.flac")])
        if file_path:
            self.uploaded_file_path = file_path
            self.predict_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Song uploaded successfully!")

    def predict(self):
        if not self.uploaded_file_path:
            messagebox.showerror("Error", "No file uploaded")
            return

        try:
            predictions = predict_(self.uploaded_file_path)
            self.display_results(predictions)
        except Exception as e:
            result_text = f"Error: {str(e)}"
            self.display_results(result_text)
    
    def display_results(self, results):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, results)
        self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = App()
    app.mainloop()
