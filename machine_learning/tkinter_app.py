import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
from predict import predict_

# Suppress warnings
warnings.filterwarnings('ignore')

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Music Genre Prediction")
        self.geometry("1000x700")  # Larger window size

        self.upload_button = tk.Button(self, text="Upload Song", command=self.upload_file, font=("Helvetica", 14))
        self.upload_button.pack(pady=20)

        self.predict_button = tk.Button(self, text="Predict Genre", command=self.predict, state=tk.DISABLED, font=("Helvetica", 16))
        self.predict_button.pack(pady=20)

        self.result_text = tk.Text(self, wrap=tk.WORD, height=5, width=100, font=("Helvetica", 12))  # Larger font
        self.result_text.pack(pady=10)
        self.result_text.config(state=tk.DISABLED)

        self.canvas = tk.Canvas(self, height=600, width=900, bg='white')
        self.canvas.pack(pady=20)

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
        genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        genre_counts = {genre: 0 for genre in genres}

        for model_data in results:
            for genre, percentage in model_data['genre'].items():
                genre_counts[genre] += int(percentage.replace('%', ''))

        most_popular_genre = max(genre_counts, key=genre_counts.get)
        most_popular_percentage = genre_counts[most_popular_genre] / len(results)

        # Display the most popular genre in the result text
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        if most_popular_percentage >= 80:
            self.result_text.insert(tk.END, f"It's {most_popular_genre} !\n(Confidence level: {most_popular_percentage:.2f}%)")
        elif most_popular_percentage >= 50:
            self.result_text.insert(tk.END, f"{most_popular_genre.capitalize()} ?\n(Confidence level: {most_popular_percentage:.2f}%)")
        else:
            self.result_text.insert(tk.END, f"It's strange to me... {most_popular_genre.capitalize()} ?\n(Confidence level: {most_popular_percentage:.2f}%)")
        self.result_text.tag_add("center", 1.0, "end")
        self.result_text.tag_configure("center", justify='center', font=("Helvetica", 24, "bold"))
        self.result_text.config(state=tk.DISABLED)
        self.canvas.delete("all")
        self.draw_histogram(results)

    def draw_histogram(self, results):
        genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        colors = ["red", "green", "blue", "orange"]

        num_models = len(results)
        canvas_height = 800
        section_height = canvas_height // num_models
        bar_width = 20
        spacing = 10

        for model_idx, model_data in enumerate(results):
            model_name = model_data['model']
            frequencies = model_data['genre']
            
            left_offset = ((model_idx)%2) * 400
            top_offset = ((model_idx)//2) * 300
            self.canvas.create_text(left_offset + 50, top_offset + 50 , text=model_name, anchor='w', fill=colors[model_idx], font=("Helvetica", 14, "bold"))

            for idx, genre in enumerate(genres):
                frequency = int(frequencies.get(genre, '0%').replace('%', ''))
                x0 = 130 + idx * (bar_width + spacing) + left_offset
                y0 = 40 + top_offset + (section_height - 50) - (frequency * (section_height - 50) // 100) 
                x1 = x0 + bar_width 
                y1 = 40 + top_offset + (section_height - 50)
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=colors[model_idx])
                self.canvas.create_text(x0 + bar_width // 2, y1 + 30, text=genre, angle=90)
                self.canvas.create_text(x0 + bar_width // 2, y0 - 10, text=f"{frequency}%", fill=colors[model_idx])

if __name__ == "__main__":
    app = App()
    app.mainloop()
