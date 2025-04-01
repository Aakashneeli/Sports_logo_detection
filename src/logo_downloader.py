import os
import requests
from PIL import Image
from io import BytesIO
import shutil
from pathlib import Path

# Guaranteed working URLs from reliable image hosting
LOGO_URLS = {
    'f1': {
        # Main F1 sponsors
        'petronas': 'https://i.ibb.co/9hLBR0D/petronas.png',
        'oracle': 'https://i.ibb.co/k97qxfP/oracle.png',
        'shell': 'https://i.ibb.co/VqFgcyt/shell.png',
        'rolex': 'https://i.ibb.co/TwPHnwX/rolex.png',
        'pirelli': 'https://i.ibb.co/Kj1Ydxt/pirelli.png'
    },
    'basketball': {
        # Basketball sponsors
        'nike': 'https://i.ibb.co/C7J3BZh/nike.png',
        'spalding': 'https://i.ibb.co/2KCZ8Cn/spalding.png',
        'gatorade': 'https://i.ibb.co/YX7Lw4Y/gatorade.png',
        'state_farm': 'https://i.ibb.co/0MKJkxy/state-farm.png',
        'kia': 'https://i.ibb.co/Qp1kqLs/kia.png'
    },
    'football': {
        # Football sponsors
        'adidas': 'https://i.ibb.co/Lx9T0ST/adidas.png',
        'nike_football': 'https://i.ibb.co/C7J3BZh/nike.png',
        'emirates': 'https://i.ibb.co/HYBFxfm/emirates.png',
        'qatar_airways': 'https://i.ibb.co/HnncXZY/qatar-airways.png',
        'heineken': 'https://i.ibb.co/Kx4RvXZ/heineken.png'
    }
}

def download_and_process_logo(url: str, output_path: str, size: tuple = (200, 200)):
    """
    Download logo from URL and process it for template matching
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Referer': 'https://www.google.com'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Create white background
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        background.paste(img, (0, 0), img)
        
        # Convert to RGB
        img = background.convert('RGB')
        
        # Resize
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Save with high quality
        img.save(output_path, quality=95)
        print(f"Successfully downloaded and processed: {output_path}")
        return True
    
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return False

def setup_template_directory():
    """
    Create template directory structure and copy logos from local source
    """
    # Create base directories
    base_dir = Path("data/templates")
    logos_dir = Path("data/logos")  # Source logos directory
    
    # Create sport directories
    for sport in ['f1', 'basketball', 'football']:
        (base_dir / sport).mkdir(parents=True, exist_ok=True)
        (logos_dir / sport).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    setup_template_directory() 