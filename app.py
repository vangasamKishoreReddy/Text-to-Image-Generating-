import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import threading

# Replace with your Hugging Face token
auth_token = "Hugging_Access_Token_Key"

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Widgets
prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = tk.Label(master=app, height=512, width=512, bg="white")
lmain.place(x=10, y=110)

# Set device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model
model_id = "_______________"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, use_auth_token=auth_token)
pipe.to(device)

# Enable xformers for faster attention computation (if available)
if torch.cuda.is_available():
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        print("xformers not installed. Install it for better performance.")

# Generate function
def generate():
    def _generate():
        trigger.configure(state=tk.DISABLED, text="Generating...")
        try:
            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    image = pipe(prompt.get(), guidance_scale=8.5, height=512, width=512, num_inference_steps=30).images[0]
            else:
                image = pipe(prompt.get(), guidance_scale=8.5, height=512, width=512, num_inference_steps=30).images[0]

            image = image.resize((512, 512))  # Resize the image
            image.save('generatedimage.png')
            img = ImageTk.PhotoImage(image)
            lmain.configure(image=img)
            lmain.image = img  # Keep reference to prevent garbage collection
        except RuntimeError as e:
            print(f"Runtime error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            trigger.configure(state=tk.NORMAL, text="Generate")

    threading.Thread(target=_generate, daemon=True).start()

# Trigger button
trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()