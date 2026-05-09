import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext

from src.feature_extractor import extract_features
from src.searcher_postgres import Searcher


class FruitSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🍎 Fruit Image Search System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize searcher
        self.searcher = None
        self.query_image_path = None
        self.query_features = None
        self.search_results = None
        
        self.setup_ui()
        self.load_searcher()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🍎 Fruit Image Search System", 
            font=("Arial", 24, "bold"),
            bg="#2c3e50", 
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main container with two columns
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Query image selection
        left_panel = tk.Frame(main_container, bg="white", relief=tk.RIDGE, bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Query Image Section
        query_label = tk.Label(
            left_panel, 
            text="Query Image", 
            font=("Arial", 14, "bold"),
            bg="white"
        )
        query_label.pack(pady=10)
        
        # Image preview frame
        self.image_frame = tk.Frame(left_panel, bg="#e0e0e0", width=280, height=250, relief=tk.SUNKEN, bd=2)
        self.image_frame.pack(padx=10, pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, bg="#e0e0e0", text="No image selected")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = tk.Frame(left_panel, bg="white")
        button_frame.pack(padx=10, pady=10, fill=tk.X)
        
        browse_btn = tk.Button(
            button_frame,
            text="📁 Browse Image",
            command=self.browse_image,
            font=("Arial", 11),
            bg="#3498db",
            fg="white",
            padx=10,
            pady=8,
            relief=tk.RAISED,
            bd=2
        )
        browse_btn.pack(fill=tk.X, pady=5)
        
        search_btn = tk.Button(
            button_frame,
            text="🔍 Search",
            command=self.perform_search,
            font=("Arial", 11, "bold"),
            bg="#27ae60",
            fg="white",
            padx=10,
            pady=8,
            relief=tk.RAISED,
            bd=2
        )
        search_btn.pack(fill=tk.X, pady=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_search,
            font=("Arial", 11),
            bg="#e74c3c",
            fg="white",
            padx=10,
            pady=8,
            relief=tk.RAISED,
            bd=2
        )
        clear_btn.pack(fill=tk.X, pady=5)
        
        # Status section
        status_label = tk.Label(
            left_panel,
            text="Status",
            font=("Arial", 12, "bold"),
            bg="white"
        )
        status_label.pack(pady=(20, 10))
        
        self.status_text = tk.Label(
            left_panel,
            text="Ready",
            font=("Arial", 10),
            bg="white",
            fg="#27ae60",
            wraplength=280,
            justify=tk.LEFT
        )
        self.status_text.pack(padx=10, pady=5, anchor=tk.W)
        
        # Initialize topk_var with fixed value of 5
        self.topk_var = tk.IntVar(value=5)
        
        # Right panel - Search results
        right_panel = tk.Frame(main_container, bg="white", relief=tk.RIDGE, bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        results_label = tk.Label(
            right_panel,
            text="Search Results",
            font=("Arial", 14, "bold"),
            bg="white"
        )
        results_label.pack(pady=10)
        
        # Results frame with scrollbar
        results_container = tk.Frame(right_panel, bg="white")
        results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(results_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(
            results_container,
            bg="white",
            yscrollcommand=scrollbar.set,
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)
        
        def on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            # Match canvas width
            self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())
        
        def on_canvas_configure(event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        
        self.scrollable_frame.bind("<Configure>", on_frame_configure)
        self.canvas.bind("<Configure>", on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def load_searcher(self):
        """Load the searcher from database"""
        try:
            self.update_status("Loading database...", "blue")
            self.root.update()
            self.searcher = Searcher()
            self.update_status("Database loaded successfully!", "green")
        except Exception as e:
            self.update_status(f"Error loading database: {str(e)}", "red")
            messagebox.showerror("Database Error", f"Failed to load database:\n{str(e)}")
    
    def browse_image(self):
        """Open file dialog to select image"""
        query_images_path = Path("dataset/query_images")
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        )
        
        file_path = filedialog.askopenfilename(
            title="Select Query Image",
            initialdir=str(query_images_path) if query_images_path.exists() else ".",
            filetypes=filetypes
        )
        
        if file_path:
            self.query_image_path = file_path
            self.display_image(file_path)
            self.update_status(f"Image loaded: {Path(file_path).name}", "green")
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            image = Image.open(image_path)
            # Resize to fit frame
            image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{str(e)}")
    
    def perform_search(self):
        """Perform similarity search"""
        if self.query_image_path is None:
            messagebox.showwarning("Warning", "Please select a query image first!")
            return
        
        if self.searcher is None:
            messagebox.showerror("Error", "Database not loaded!")
            return
        
        try:
            self.update_status("Extracting features...", "blue")
            self.root.update()
            
            # Extract features
            self.query_features = extract_features(self.query_image_path)
            
            if self.query_features is None:
                self.update_status("Failed to extract features from image", "red")
                messagebox.showerror("Error", "Failed to extract features from image")
                return
            
            self.update_status("Searching database...", "blue")
            self.root.update()
            
            # Perform search
            top_k = self.topk_var.get()
            self.search_results = self.searcher.search(self.query_features, top_k=top_k)
            
            self.update_status(f"Found {len(self.search_results)} results", "green")
            self.display_results()
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")
            messagebox.showerror("Search Error", f"An error occurred:\n{str(e)}")
    
    def display_results(self):
        """Display search results in 3x2 grid layout"""
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        if not self.search_results:
            no_results_label = tk.Label(
                self.scrollable_frame,
                text="No results found",
                font=("Arial", 12),
                bg="white",
                fg="#7f8c8d"
            )
            no_results_label.pack(pady=20)
            return
        
        # Create a grid frame to hold the cards
        grid_frame = tk.Frame(self.scrollable_frame, bg="white")
        grid_frame.pack(expand=True, padx=20)
        
        # Configure grid for 3 columns
        for col in range(3):
            grid_frame.grid_columnconfigure(col, weight=0, minsize=200)
        
        row = 0
        col = 0
        
        for rank, (distance, img_path, label) in enumerate(self.search_results, 1):
            result_card = tk.Frame(grid_frame, bg="#ecf0f1", relief=tk.RAISED, bd=1)
            result_card.grid(row=row, column=col, pady=8, padx=5, sticky="nsew")
            
            # Rank and label
            header_frame = tk.Frame(result_card, bg="#34495e")
            header_frame.pack(fill=tk.X)
            
            info_text = f"#{rank} - {label}"
            info_label = tk.Label(
                header_frame,
                text=info_text,
                font=("Arial", 10, "bold"),
                bg="#34495e",
                fg="white",
                justify=tk.CENTER,
                wraplength=200
            )
            info_label.pack(anchor=tk.CENTER, padx=10, pady=8)
            
            # Image and details
            content_frame = tk.Frame(result_card, bg="white")
            content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Try to display result image
            try:
                if Path(img_path).exists():
                    result_image = Image.open(img_path)
                    result_image.thumbnail((80, 80), Image.Resampling.LANCZOS)
                    result_photo = ImageTk.PhotoImage(result_image)
                    
                    img_label = tk.Label(content_frame, image=result_photo, bg="white")
                    img_label.image = result_photo
                    img_label.pack(pady=10)
            except:
                pass
            
            # Details
            details_frame = tk.Frame(content_frame, bg="white")
            details_frame.pack(fill=tk.BOTH, expand=True)
            
            path_label = tk.Label(
                details_frame,
                text=f"Path: {Path(img_path).name}",
                font=("Arial", 8),
                bg="white",
                fg="#7f8c8d",
                wraplength=200,
                justify=tk.CENTER
            )
            path_label.pack(anchor=tk.CENTER, pady=2)
            
            # Distance info
            distance_label = tk.Label(
                details_frame,
                text=f"Distance: {distance:.4f}",
                font=("Arial", 8),
                bg="white",
                fg="#7f8c8d"
            )
            distance_label.pack(anchor=tk.CENTER, pady=2)
            
            # Similarity bar
            similarity = max(0, (1 - min(distance, 1.0))) * 100
            bar_frame = tk.Frame(details_frame, bg="white")
            bar_frame.pack(anchor=tk.CENTER, fill=tk.X, pady=5, padx=5)
            
            tk.Label(
                bar_frame,
                text="Similarity:",
                font=("Arial", 8),
                bg="white"
            ).pack(side=tk.LEFT)
            
            progress_bar = ttk.Progressbar(
                bar_frame,
                length=120,
                mode='determinate',
                value=similarity
            )
            progress_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            similarity_label = tk.Label(
                bar_frame,
                text=f"{similarity:.1f}%",
                font=("Arial", 8),
                bg="white"
            )
            similarity_label.pack(side=tk.LEFT, padx=3)
            
            # Move to next grid position
            col += 1
            if col >= 3:
                col = 0
                row += 1
    
    def clear_search(self):
        """Clear search results"""
        self.query_image_path = None
        self.query_features = None
        self.search_results = None
        
        self.image_label.config(image="", text="No image selected")
        self.image_label.image = None
        
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.update_status("Cleared", "green")
    
    def update_status(self, message, color="black"):
        """Update status message"""
        self.status_text.config(text=message, fg=color)


def main():
    root = tk.Tk()
    app = FruitSearchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
