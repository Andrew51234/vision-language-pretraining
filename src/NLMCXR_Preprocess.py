'''
This script preprocesses the NLMCXR dataset by performing the following steps:
1. Extracts sections ('COMPARISON', 'INDICATION', 'FINDINGS', 'IMPRESSION') from XML reports
2. Cleans and tokenizes the extracted text using the Stanza NLP library
3. Resizes associated image files to a specified size
4. Saves the preprocessed data to a pickle file

Functions:
    get_image_paths(): Get all PNG image paths from the dataset
    get_report_paths(): Get all report paths from the dataset
    parse_xml_files(): Parse XML files and extract relevant information
    resize_image(): Resize an image to specified dimensions
    resize_and_save_images(): Resize and save images to a new directory
    tokenize_sentences(): Tokenize text into sentences using Stanza
'''

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig
import glob
import os
import numpy as np
import re
import pickle
import pandas as pd
import xml.etree.ElementTree as ET
from pandarallel import pandarallel
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import stanza

from src import utils

log = utils.get_pylogger(__name__)

#log starting of the script
print("Starting the script")

def get_image_paths(nlmcxr_dir):
    """Get all PNG image paths from the NLMCXR dataset"""
    image_paths = glob.glob(f'{nlmcxr_dir}/**/*.png', recursive=True) 
    filtered_paths = [path for path in image_paths if 'resized' not in path.lower()]
    return sorted(filtered_paths)

def get_report_paths(reports_dir):
    """Get all report paths from the NLMCXR dataset"""
    report_paths = glob.glob(f'{reports_dir}/**/*.xml', recursive=True)
    return sorted(report_paths)

def parse_xml_files(report_paths, png_dir):
    """Parse XML files and extract relevant information into a DataFrame"""
    data = []
    for xml_file in tqdm(report_paths):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            comparison = ""
            indication = ""
            findings = ""
            impression = ""
            
            abstract = root.find(".//Abstract")
            if abstract is not None:
                for abstract_text in abstract.findall("AbstractText"):
                    label = abstract_text.get("Label")
                    text_value = abstract_text.text
                    
                    if text_value == "None.":
                        text_value = None
                    
                    if label == "COMPARISON":
                        comparison = text_value
                    elif label == "INDICATION": 
                        indication = text_value
                    elif label == "FINDINGS":
                        findings = text_value
                    elif label == "IMPRESSION":
                        impression = text_value

            for parent_image in root.findall(".//parentImage"):
                image_id = parent_image.get("id", "")
                caption = parent_image.find("caption")
                caption_text = caption.text or None
                image_path = os.path.join(png_dir, image_id+'.png')
                
                data.append({
                    'file_path': xml_file,
                    'comparison': comparison,
                    'indication': indication,
                    'findings': findings,
                    'impression': impression,
                    'image_file_path': image_path,
                    'image_caption': caption_text
                })
                
        except Exception as e:
            log.error(f"Error processing {xml_file}: {str(e)}")
            
    return pd.DataFrame(data)

def resize_image(img_path: str, max_size: int=512) -> Image:
    """Resize the image to have a max size while maintaining aspect ratio"""
    try:
        img_copy = Image.open(img_path).copy()
        img_copy.thumbnail((max_size, max_size))
        return img_copy
    except (UnidentifiedImageError, OSError) as e:
        log.error(f"Skipping corrupt image file: {img_path}")
        return None

def resize_and_save_images(row, nlmcxr_dir, target_size=512):
    """Resize and save images to a new directory"""
    img_path = row['image_file_path']
    relative_path = os.path.relpath(img_path, os.path.join(nlmcxr_dir, 'NLMCXR_png'))
    resized_dir = os.path.join(nlmcxr_dir, 'NLMCXR_png_resized', os.path.dirname(relative_path))
    os.makedirs(resized_dir, exist_ok=True)
    
    resized_image_path = os.path.join(resized_dir, 
                                     f"{os.path.splitext(os.path.basename(img_path))[0]}_resized.png")
    
    # Accessing images from the script returns a permission error for some reason.. we can skip this for now since we already have the images resized
    
    # if not os.path.exists(resized_image_path):
    #     resized_image = resize_image(img_path, target_size)
    #     if resized_image is not None:
    #         try:
    #             resized_image.save(resized_image_path)
    #         except PermissionError:
    #             log.error(f"Permission denied when saving to {resized_image_path}")
    #             return None
    
    return resized_image_path

def tokenize_sentences(text: str, nlp) -> list:
    """Tokenize text into sentences using Stanza"""
    if not isinstance(text, str) or not text.strip():
        return []

    doc = nlp(text)
    tokenized_words = []
    for sentence in doc.sentences:
        tokenized_words.extend([word.text for word in sentence.words])
    
    return tokenized_words

@hydra.main(version_base=None, config_path='../configs', config_name="nlmcxr_preprocessing")
def main(cfg: DictConfig) -> None:
    # Initialize pandarallel
    pandarallel.initialize(progress_bar=True)

    # Setup paths
    NLMCXR_DIR = cfg.paths.nlmcxr_dir
    NLMCXR_PNG_DIR = os.path.join(NLMCXR_DIR, 'images')
    NLMCXR_REPORTS_DIR = os.path.join(NLMCXR_DIR, 'texts/ecgen-radiology')
    STANZA_DIR = os.path.join(cfg.paths.stanza_dir, 'stanza_en')

    log.info("Verifying directories...")
    log.info(f"NLMCXR directory exists: {os.path.exists(NLMCXR_DIR)}")
    log.info(f"NLMCXR PNG directory exists: {os.path.exists(NLMCXR_PNG_DIR)}")
    log.info(f"NLMCXR REPORTS directory exists: {os.path.exists(NLMCXR_REPORTS_DIR)}")
    log.info(f"STANZA directory exists: {os.path.exists(STANZA_DIR)}")
    # Get image and report paths
    image_paths = get_image_paths(NLMCXR_DIR)
    report_paths = get_report_paths(NLMCXR_REPORTS_DIR)
    log.info(f"Found {len(image_paths)} images and {len(report_paths)} reports")

    # Parse XML files
    log.info("Parsing XML files...")
    nlmcxr_df = parse_xml_files(report_paths, NLMCXR_PNG_DIR)

    # Remove empty samples
    mask_to_drop = (nlmcxr_df['comparison'].isnull() & 
                    nlmcxr_df['indication'].isnull() & 
                    nlmcxr_df['findings'].isnull() & 
                    nlmcxr_df['impression'].isnull())
    
    log.info(f"Removing {mask_to_drop.sum()} samples with all report fields empty")
    nlmcxr_df = nlmcxr_df[~mask_to_drop].reset_index(drop=True)

    # Resize images
    log.info("Resizing images...")
    for _, row in tqdm(nlmcxr_df.iterrows(), total=len(nlmcxr_df)):
        resized_path = resize_and_save_images(row, NLMCXR_DIR)
        nlmcxr_df.loc[row.name, 'image_file_path_resized'] = resized_path

    # Initialize Stanza and tokenize text
    log.info("Tokenizing text...")
    nlp = stanza.Pipeline(lang='en', processors='tokenize', model_dir=STANZA_DIR)

    for field in ['comparison', 'indication', 'findings', 'impression', 'image_caption']:
        nlmcxr_df[f'{field.lower()}_tokenized'] = nlmcxr_df[field].apply(
            lambda x: tokenize_sentences(x, nlp)
        )

    # Drop original text columns
    columns_to_drop = ['comparison', 'indication', 'findings', 'impression', 
                      'image_caption', 'image_file_path']
    nlmcxr_df.drop(columns=columns_to_drop, inplace=True)

    # Save preprocessed data
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.paths.output_dir, 'nlmcxr_preprocessed.pkl')
    log.info(f"Saving preprocessed data to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(nlmcxr_df, f)

    log.info("Preprocessing completed successfully")

if __name__ == "__main__":
    main()