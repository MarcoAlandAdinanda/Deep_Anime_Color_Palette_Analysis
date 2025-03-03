import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from sklearn.cluster import KMeans

class ColorPaletteExtractor:
    def __init__(self):
        self.image = None
        self.palette = None

    def load_image_from_url(self, url):
        """Load an image from a URL."""
        try:
            resp = urllib.request.urlopen(url)
            image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            self.image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if self.image is None:
                raise ValueError("Failed to decode image from URL")
        except Exception as e:
            raise RuntimeError(f"Error loading image from URL: {str(e)}")

    def load_image_from_file(self, filepath):
        """Load an image from a local file path."""
        self.image = cv2.imread(filepath)
        if self.image is None:
            raise ValueError(f"Failed to load image from file: {filepath}")

    def extract_palette(self, num_clusters=5):
        """Extract dominant colors from the loaded image using K-Means clustering."""
        if self.image is None:
            raise RuntimeError("No image loaded. Load an image first.")
        
        # Convert image and prepare for processing
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape((-1, 3))
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(pixels)
        
        # Process and store results
        dominant_colors = np.array(kmeans.cluster_centers_, dtype=np.uint8)
        
        # Calculate luminance for sorting (using RGB values)
        luminance = dominant_colors[:, 0] * 0.299 + dominant_colors[:, 1] * 0.587 + dominant_colors[:, 2] * 0.114
        sorted_indices = np.argsort(luminance)[::-1]  # Sort descending
        
        # Sort colors by luminance
        dominant_colors_sorted = dominant_colors[sorted_indices]
        
        self.palette = cv2.cvtColor(
            dominant_colors_sorted.reshape(1, -1, 3), 
            cv2.COLOR_RGB2BGR
        ).reshape(-1, 3)

    def display_palette(self):
        """Display the extracted color palette."""
        if self.palette is None:
            raise RuntimeError("No palette extracted. Extract palette first.")
        
        # Create visualization
        patch = np.zeros((50, 300, 3), dtype=np.uint8)
        num_colors = len(self.palette)
        patch_width = 300 // num_colors
        
        for i, color in enumerate(self.palette):
            start_col = i * patch_width
            end_col = (i + 1) * patch_width
            patch[:, start_col:end_col] = color
        
        # Convert and display
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        plt.imshow(patch_rgb)
        plt.title("Dominant Color Palette (Sorted by Luminance)")
        plt.axis("off")
        plt.show()

    def get_palette(self):
        """Return the extracted color palette."""
        return self.palette.copy() if self.palette is not None else None

if __name__ == "__main__":
    extractor = ColorPaletteExtractor()
    try:
        # url = "https://cdn.myanimelist.net/images/characters/5/496454.jpg"
        url1 = "https://cdn.myanimelist.net/images/characters/15/422168.jpg"
        # url2 = "https://cdn.myanimelist.net/images/characters/12/514093.jpg"
        extractor.load_image_from_url(url1)
        extractor.extract_palette(num_clusters=30)
        extractor.display_palette()
        print("Extracted Palette (BGR format):")
        print(extractor.get_palette())
    except Exception as e:
        print(f"Error: {str(e)}")