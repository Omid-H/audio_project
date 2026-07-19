import os
import requests
from tqdm import tqdm

# Define the dictionary of models and their download URLs
# Replace "URL_HERE" with actual direct download links when they are available
MODELS = {
    # ------ مدل‌های پایه ------
    "rvc_models/hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
    "rvc_models/rmvpe.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    
    # ------ مدل باب اسفنجی ------
    "rvc_models/BabAsfanj/Bab_Asfanj.zip": "https://huggingface.co/PlushymehereJC/Spongebob_Persian_dub/resolve/main/Bab_Asfanj.zip",
    
    # ------ مدل مورگان فریمن (جدید) ------
    "rvc_models/MorganFreeman/Morgan_Freeman.zip": "https://huggingface.co/DeputyRipper/Morgan_Freeman_RVCV2/resolve/main/Morgan%20Freeman.zip"
}

def download_file(url: str, dest_path: str) -> bool:
    """
    Downloads a file from a URL to a target destination with a progress bar.
    
    Parameters:
        url (str): Direct download link.
        dest_path (str): Destination path on local system.
        
    Returns:
        bool: True if downloaded or already present, False if download failed.
    """
    # Smart Check: If the file exists and is not empty, skip downloading
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"Present: '{os.path.basename(dest_path)}' is already downloaded.")
        return True

    # Check if a valid URL has been provided or if it's still a placeholder
    if not url or url == "URL_HERE":
        print(f"Skipped: No download URL configured for '{os.path.basename(dest_path)}'.")
        return False

    # Create the parent directory structure if it doesn't already exist
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    print(f"Downloading '{os.path.basename(dest_path)}'...")
    
    # We download to a temporary file first and rename it only when complete.
    # This prevents corrupted partially-downloaded files from being recognized as complete.
    temp_path = dest_path + ".tmp"
    
    try:
        # Stream the download request to handle large files without high memory usage
        response = requests.get(url, stream=True, timeout=60)
        # Raise an exception for bad status codes (4xx, 5xx)
        response.raise_for_status()
        
        # Extract total content length if provided by the server
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB chunks for large file optimization
        
        with open(temp_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=block_size):
                size = file.write(data)
                progress_bar.update(size)
        
        # Atomically rename/replace the temporary file to the final destination
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        print(f"Success: '{os.path.basename(dest_path)}' downloaded successfully.")
        return True
        
    except requests.exceptions.RequestException as req_err:
        print(f"Network error downloading '{os.path.basename(dest_path)}': {req_err}")
    except IOError as io_err:
        print(f"File writing error for '{os.path.basename(dest_path)}': {io_err}")
    except Exception as err:
        print(f"Unexpected error occurred for '{os.path.basename(dest_path)}': {err}")
    
    # Clean up the temporary file if download failed
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
            
    return False

def main():
    """
    Main orchestrator that iterates over the defined models, resolves their destination paths
    relative to the script directory, and downloads them if needed.
    """
    # Resolve paths relative to the directory where this script resides to support execution from any working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("Starting Setup & Download of Large AI Models")
    print("=" * 60)
    
    downloaded_count = 0
    total_models = len(MODELS)
    
    for rel_path, url in MODELS.items():
        # Handle cross-platform path separators (converts forward slashes to match current OS)
        normalized_path = os.path.join(*rel_path.split("/"))
        dest_path = os.path.abspath(os.path.join(script_dir, normalized_path))
        
        if download_file(url, dest_path):
            downloaded_count += 1
            
    print("-" * 60)
    print(f"Model setup finished. {downloaded_count}/{total_models} models verified/ready.")
    print("=" * 60)

if __name__ == "__main__":
    main()
