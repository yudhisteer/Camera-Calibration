import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pyautogui
import time

class CheckerboardDisplay:
    def __init__(self, checkerboard_size=(7,7), square_size=100):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
    def generate_checkerboard(self):
        """Generate a checkerboard pattern"""
        rows, cols = self.checkerboard_size
        board_rows = rows + 1
        board_cols = cols + 1
        
        # Create the checkerboard image
        img_size = (self.square_size * board_rows, self.square_size * board_cols)
        img = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        
        # Fill in the squares
        for i in range(board_rows):
            for j in range(board_cols):
                if (i + j) % 2 == 0:
                    x1 = j * self.square_size
                    y1 = i * self.square_size
                    x2 = (j + 1) * self.square_size
                    y2 = (i + 1) * self.square_size
                    img[y1:y2, x1:x2] = 255
        
        return img

    def display_and_capture(self):
        """Display the checkerboard pattern and capture the screen automatically"""
        # Generate the checkerboard
        img = self.generate_checkerboard()
        
        # Create tkinter window
        root = tk.Tk()
        
        # Set window to fullscreen immediately
        root.attributes('-fullscreen', True)
        
        # Convert OpenCV image to PIL format
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Create label and display image
        label = tk.Label(root, image=img_tk)
        label.pack(fill='both', expand=True)
        
        def auto_capture():
            # Wait for window to be fully displayed
            root.update()
            time.sleep(2)  # Give time for the window to settle
            
            # Capture the screen
            screenshot = pyautogui.screenshot()
            screenshot.save('reference_screen.png')
            print("Screen captured and saved as 'reference_screen.png'")
            
            # Close the window
            root.quit()
        
        # Schedule the capture to happen automatically
        root.after(100, auto_capture)
        
        root.mainloop()
        root.destroy()

if __name__ == "__main__":
    display = CheckerboardDisplay(checkerboard_size=(7,7), square_size=100)
    display.display_and_capture()