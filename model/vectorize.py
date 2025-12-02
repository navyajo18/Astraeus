from PIL import Image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def get_images(classfication):
    script_dir = os.path.dirname(__file__)
    asteroids_dir = os.path.join(script_dir, '..', 'data', classfication)
    asteroids_vectorized_dir = os.path.join(script_dir, '..', 'data', f"{classfication}_vectorized")
    all_image_vectors = [] 
    if not os.path.isdir(asteroids_dir):
        print(f"Error: Directory not found at {asteroids_dir}")
    else:
        for filename in os.listdir(asteroids_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_full_path = os.path.join(asteroids_dir, filename)
                image = mpimg.imread(image_full_path)
                img_array = np.array(image)
                base_filename = os.path.splitext(filename)[0] 
                output_filename = f"{base_filename}.npy"
                output_full_path = os.path.join(asteroids_vectorized_dir, output_filename)
                if os.path.exists(output_full_path):
                    print(f"Skipping {filename}")
                    continue 
                np.save(output_full_path, img_array)
get_images('asteroids')
get_images('black_holes')
get_images('planets')
get_images('stars')