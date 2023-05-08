import tkinter as tk
from tkinter import filedialog
import subprocess

def read_yaml_file():
    file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return file_path, content

def save_yaml_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

def show_file_content():
    global current_file_path
    current_file_path, content = read_yaml_file()
    text.delete(1.0, tk.END)
    text.insert(tk.INSERT, content)

def save_and_run():
    content = text.get(1.0, tk.END)
    save_yaml_file(current_file_path, content)
    command = f"python {button_to_command[current_button]} {current_file_path}"
    subprocess.run(command, shell=True)

def save_file_as():
    content = text.get(1.0, tk.END)
    new_file_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
    if new_file_path:
        save_yaml_file(new_file_path, content)
        command = f"python {button_to_command[current_button]} {new_file_path}"
        subprocess.run(command, shell=True)
    
app = tk.Tk()
app.title("配置文件编辑器")

current_file_path = None
current_button = None
button_to_command = {
    "Train Model": "train_model.py",
    "Plot Result": "plot_result.py",
    "Update Data": "update_data.py",
    "Test Model": "test_model.py",
    "Deploy Model": "deploy_model.py",
}

def on_click(button_text):
    global current_button
    current_button = button_text
    show_file_content()

frame = tk.Frame(app)
frame.pack()

for button_text in button_to_command.keys():
    button = tk.Button(frame, text=button_text, command=lambda text=button_text: on_click(text))
    button.pack(side=tk.LEFT)

text = tk.Text(app, wrap=tk.WORD)
text.pack()

save_and_run_button = tk.Button(app, text="保存并运行", command=save_and_run)
save_and_run_button.pack()

save_as_button = tk.Button(app, text="文件另存并执行", command=save_file_as)
save_as_button.pack()

app.mainloop()