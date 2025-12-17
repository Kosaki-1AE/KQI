import os
import tkinter as tk
from tkinter import filedialog


def embed_invisible_url(text, url):
    invisible = '\u200B'
    hidden_url = invisible.join(url)
    return text+hidden_url

def generate_file():
    text = text_entry.get("1.0", tk.END).strip()
    url = url_entry.get().strip()

    if not text or not url:
        result_label.config(text="文章とURLを両方入力してな！", fg="red")
        return

    message = embed_invisible_url(text,url)

    filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")], initialfile="250502_不可視URL付き文章.txt")

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(message)
        result_label.config(text=f"保存完了！\n{os.path.basename(filepath)} に書き出したで！", fg="blue")

# GUI
root = tk.Tk()
root.title("不可視URL埋め込みメーカー")
root.geometry("500x400")

tk.Label(root, text="文章を入力：").pack()
text_entry = tk.Text(root, height=7, width=60)
text_entry.pack()

tk.Label(root, text="埋め込むURL：").pack()
url_entry = tk.Entry(root, width=60)
url_entry.pack()

tk.Button(root, text="不可視URL付きで保存！", command=generate_file).pack(pady=10)

result_label = tk.Label(root, text="", fg="blue")
result_label.pack()

root.mainloop()
