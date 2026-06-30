import os
import urllib.request
import time
import numpy as np
import cv2
import random

def simulate_screen_recapture(img_np):
    h, w, c = img_np.shape
    
    # 1. Apply a pixel grid pattern (subpixels)
    y_pitch = random.uniform(3.0, 6.0)
    x_pitch = random.uniform(3.0, 6.0)
    
    ys = np.arange(h)
    xs = np.arange(w)
    X, Y = np.meshgrid(xs, ys)
    
    grid_r = (0.5 + 0.5 * np.cos(2 * np.pi * X / x_pitch)) * (0.5 + 0.5 * np.cos(2 * np.pi * Y / y_pitch))
    grid_g = (0.5 + 0.5 * np.cos(2 * np.pi * (X - x_pitch/3) / x_pitch)) * (0.5 + 0.5 * np.cos(2 * np.pi * Y / y_pitch))
    grid_b = (0.5 + 0.5 * np.cos(2 * np.pi * (X - 2*x_pitch/3) / x_pitch)) * (0.5 + 0.5 * np.cos(2 * np.pi * Y / y_pitch))
    grid_rgb = np.stack([grid_r, grid_g, grid_b], axis=-1)
    
    # 2. Add Moire interference pattern
    moire_noise = np.zeros((h, w, 3))
    num_bands = random.randint(1, 3)
    for _ in range(num_bands):
        angle = random.uniform(0, np.pi)
        freq = random.uniform(0.015, 0.05)
        phase = random.uniform(0, 2*np.pi)
        for channel in range(3):
            ch_phase = phase + random.uniform(-0.5, 0.5)
            ch_wave = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (X * np.cos(angle) + Y * np.sin(angle)) + ch_phase)
            moire_noise[..., channel] += ch_wave * 0.15
            
    moire_noise = moire_noise / num_bands
    
    # 3. Simulate glare (screen reflection)
    cx = random.uniform(0, w)
    cy = random.uniform(0, h)
    sigma = random.uniform(w * 0.3, w * 0.8)
    glare = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    glare_rgb = np.stack([glare, glare, glare], axis=-1) * random.uniform(0.05, 0.25)
    
    # 4. Integrate components
    sampled = img_np.astype(float) / 255.0
    grid_blend = random.uniform(0.15, 0.35)
    sampled = sampled * (1.0 - grid_blend + grid_blend * grid_rgb)
    sampled = sampled + moire_noise + glare_rgb
    
    # 5. Apply non-linear contrast (sensor capture curve)
    gamma = random.uniform(1.1, 1.6)
    sampled = np.clip(sampled, 0.0, 1.0)
    sampled = np.power(sampled, gamma)
    
    # 6. Blur slightly (simulating lens focus)
    blur_k = random.choice([3, 5])
    sampled_uint8 = (sampled * 255.0).astype(np.uint8)
    sampled_uint8 = cv2.GaussianBlur(sampled_uint8, (blur_k, blur_k), 0)
    
    return sampled_uint8

def main():
    real_dir = "real"
    screen_dir = "screen"
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(screen_dir, exist_ok=True)
    
    print("Downloading 50 real photos and generating recaptures...")
    
    num_downloaded = 0
    total_images = 50
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    # Let's download a variety of images (faces, scenes, text, documents)
    # Using Picsum Photos with different seeds/endpoints
    for i in range(total_images):
        seed = 1000 + i
        url = f"https://picsum.photos/id/{i}/800/800"
        # If ID endpoints don't work, use random endpoints with seeds
        if i % 5 == 0:
            url = f"https://picsum.photos/800/800?random={i}"
            
        real_path = os.path.join(real_dir, f"im_{i:03d}.jpg")
        screen_path = os.path.join(screen_dir, f"im_{i:03d}.jpg")
        
        success = False
        retries = 3
        while retries > 0 and not success:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = response.read()
                    with open(real_path, 'wb') as f:
                        f.write(data)
                
                # Check if download is valid image
                img = cv2.imread(real_path)
                if img is not None and img.shape[0] >= 100:
                    # Successfully downloaded, now simulate recapture
                    screen_img = simulate_screen_recapture(img)
                    cv2.imwrite(screen_path, screen_img)
                    success = True
                    num_downloaded += 1
                    print(f"Downloaded and processed image {num_downloaded}/{total_images} (size: {img.shape})")
                else:
                    print(f"Fail to read image from {url}, retrying...")
                    retries -= 1
                    time.sleep(1)
            except Exception as e:
                print(f"Error downloading {url}: {e}, retrying...")
                retries -= 1
                time.sleep(1)
                
        # Sleep a little to be polite to the host
        time.sleep(0.5)
        
    print(f"Finished! Successfully built dataset with {num_downloaded} real and {num_downloaded} screen recapture images.")

if __name__ == "__main__":
    main()
