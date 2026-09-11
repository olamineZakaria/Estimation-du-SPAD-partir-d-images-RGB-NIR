import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import random

DIRS = ["nir", os.path.join("cler", "rgb")]


def get_key(filename: str) -> str:
    return filename.split("_Tricam")[0]


class FourPointMaskTk:
    def __init__(self, root):
        self.root = root
        self.root.title("4-Point Mask Tool")

        self.display_w = 900
        self.display_h = 650

        self.dir_index = 0
        self.current_dir = None
        self.files = []

        self.image = None
        self.image_disp = None
        self.h = None
        self.w = None

        self.filename = None
        self.filepath = None  # full path of currently loaded image
        self.key = None

        self.points = []

        # Canvas
        self.canvas = tk.Canvas(root, width=self.display_w, height=self.display_h, bg="black")
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.click)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Open Image...", command=self.load_from_file).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Next (R)", command=self.load_random).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Reset Points", command=self.reset_points).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Quit", command=root.quit).pack(side=tk.LEFT)

        # status label to show current file
        self.status_label = tk.Label(root, text="No image loaded", anchor="w")
        self.status_label.pack(fill=tk.X)

        root.bind("r", lambda e: self.load_random())
        root.bind("o", lambda e: self.load_from_file())
        root.bind("q", lambda e: root.quit())

        # Start with no image; user can pick "Open Image..." or "Next (R)"
        self.update_status()

    # ---------------- LOAD IMAGE (FROM LOCAL FILE) ----------------
    def load_from_file(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return  # user cancelled

        self._load_image_from_path(path)

    def _load_image_from_path(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not open image:\n{path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.image = img
        self.h, self.w = img.shape[:2]

        self.filepath = path
        self.filename = os.path.basename(path)
        self.current_dir = os.path.dirname(path)

        self.points = []
        self.draw_image()
        self.update_status()

    # ---------------- LOAD RANDOM (ORIGINAL BEHAVIOR) ----------------
    def load_random(self):
        if self.dir_index >= len(DIRS):
            print("DONE: all directories processed.")
            return

        current_dir = DIRS[self.dir_index]
        if not os.path.isdir(current_dir):
            self.dir_index += 1
            self.load_random()
            return

        files = [
            f for f in os.listdir(current_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not files:
            self.dir_index += 1
            self.load_random()
            return

        if self.dir_index == 0:
            fname = random.choice(files)
            self.key = get_key(fname)
        else:
            matches = [f for f in files if get_key(f) == self.key]
            fname = random.choice(matches) if matches else random.choice(files)

        self.current_dir = current_dir
        path = os.path.join(current_dir, fname)
        self._load_image_from_path(path)

    # ---------------- DISPLAY ----------------
    def draw_image(self):
        if self.image is None:
            return

        img = self.image.copy()

        # draw points + lines
        for p in self.points:
            cv2.circle(img, p, 5, (255, 0, 0), -1)

        for i in range(1, len(self.points)):
            cv2.line(img, self.points[i - 1], self.points[i], (255, 255, 0), 2)

        if len(self.points) == 4:
            cv2.line(img, self.points[3], self.points[0], (0, 255, 0), 2)

        # resize for display
        img = cv2.resize(img, (self.display_w, self.display_h))

        # convert to Tkinter format
        self.photo = ImageTk.PhotoImage(Image.fromarray(img))

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def update_status(self):
        if self.filepath:
            self.status_label.config(text=f"Loaded: {self.filepath}  |  points: {len(self.points)}/4")
        else:
            self.status_label.config(text="No image loaded")

    # ---------------- CLICK ----------------
    def click(self, event):
        if self.image is None or len(self.points) >= 4:
            return

        ix = int(event.x * self.w / self.display_w)
        iy = int(event.y * self.h / self.display_h)

        self.points.append((ix, iy))
        self.draw_image()
        self.update_status()

        if len(self.points) == 4:
            self.save()

    def reset_points(self):
        self.points = []
        self.draw_image()
        self.update_status()

    # ---------------- MASK ----------------
    def generate_mask(self):
        mask = np.zeros((self.h, self.w), dtype=np.uint8)

        if len(self.points) == 4:
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        return mask

    # ---------------- SAVE (CHOOSE DESTINATION) ----------------
    def save(self):
        mask = self.generate_mask()

        default_name = f"mask_{self.filename}" if self.filename else "mask.png"
        default_dir = self.current_dir if self.current_dir and os.path.isdir(self.current_dir) else os.getcwd()

        save_path = filedialog.asksaveasfilename(
            title="Save mask as...",
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )

        if not save_path:
            # user cancelled the save dialog; keep points so they can retry saving
            print("Save cancelled by user.")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask)
        print("Saved:", save_path)
        messagebox.showinfo("Saved", f"Mask saved to:\n{save_path}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FourPointMaskTk(root)
    root.mainloop()