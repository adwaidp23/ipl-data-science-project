import os
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_extract_data(url: str, download_path: str, extract_to: str):
    """
    Downloads a zip file from a URL and extracts it to a destination folder.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    logging.info(f"Downloading data from {url}...")
    try:
        urllib.request.urlretrieve(url, download_path)
        logging.info("Download completed successfully.")
        
        logging.info(f"Extracting data to {extract_to}...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info("Extraction completed successfully.")
        
        # Clean up the zip file
        os.remove(download_path)
        logging.info("Removed temporary zip file.")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Cricsheet IPL data in CSV format
    CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    ZIP_PATH = os.path.join(BASE_DIR, "data", "ipl_data.zip")
    
    download_and_extract_data(CRICSHEET_URL, ZIP_PATH, DATA_DIR)
