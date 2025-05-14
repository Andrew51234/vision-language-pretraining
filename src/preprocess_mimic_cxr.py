'''
This script preprocesses the MIMIC-CXR dataset by performing the following steps:
1. Extracts 'FINDINGS' and 'IMPRESSION' sections from text reports.
2. Cleans and tokenizes the extracted text using the Stanza NLP library.
3. Finds and resizes associated image files.
4. Saves the preprocessed data to a specified output directory.
Functions:
    get_findings_impressions(row, path):
    clean_text(sentence: stanza.models.common.doc.Sentence):
        Removes new lines and leading colons from the sentence.
    get_sentences(document: stanza.Document, min_words=3):
        Extracts sentences from a Stanza document, filtering out short sentences.
    generate_image_path(row, mimic_cxr_jpg):
        Generates image file paths and checks their existence.
    resize_image(img_path: str, max_size: int=256) -> Image:
        Resizes the image to have a size of max_size on the larger side.
    resize_and_save_images(row, base_directory, target_size=256):
Main Function:
    main(cfg: DictConfig) -> None:
        The main function that orchestrates the preprocessing steps using configurations provided by Hydra.
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
import pandas as pd
from pandarallel import pandarallel
import stanza
from tqdm import tqdm
from PIL import Image

from src import utils

log = utils.get_pylogger(__name__)

def get_findings_impressions(row, path):
    """
    Extracts the 'FINDINGS' and 'IMPRESSION' sections from a text file corresponding to a patient's study.
    Args:
        df (pd.DataFrame): A DataFrame containing patient information with columns 'patient_folder', 'patient_id', and 'study_id'.
        path (str): The base directory path where the patient files are stored.
    Returns:
        tuple: A tuple containing two strings:
            - findings (str): The extracted 'FINDINGS' section from the text file.
            - impression (str): The extracted 'IMPRESSION' section from the text file.
    """

    findings, impression = '', ''
    with open(f'{path}/files/{row["patient_folder"]}/{row["patient_id"]}/{row["study_id"]}.txt') as f:
        data = f.read()
        # Check for both FINDINGS and impression
        matches = re.search(r"^([\w\W]+?)\bFINDINGS\b([\w\W]+?)\bIMPRESSION\b([\w\W]+?)$", data)
        if matches and len(matches.groups())==3:
            findings = matches.group(2)
            impression = matches.group(3)
        if len(findings)==0:
            findings_match = re.search(r"^([\w\W]+?)\bFINDINGS\b([\w\W]+?)$", data)
            if findings_match and len(findings_match.groups())==2:
                findings = findings_match.group(2)
        if len(impression)==0:
            impression_match = re.search(r"^([\w\W]+?)\bIMPRESSION\b([\w\W]+?)$", data)
            #log.info(len(impression_match.groups()))
            if impression_match and len(impression_match.groups())==2:
                impression = impression_match.group(2)

    return findings, impression

def clean_text(sentence: stanza.models.common.doc.Sentence):
    '''
    Remove new lines and leading colons from the sentence
    '''
    final_sentence = sentence.text.replace('\n', '').strip()
    if final_sentence.startswith(":"):
        final_sentence = final_sentence[1:].strip()

    return final_sentence

def get_sentences(document: stanza.Document, min_words=3):
    '''
    Extract sentences from a Stanza document, filtering out short sentences
    and cleaning the text
    '''
    final_sentences = []
    for sentence in document.sentences:
        if len(sentence.words) < min_words:
            continue
        final_sentences.append(clean_text(sentence))
    return final_sentences

def generate_image_path(row, mimic_cxr_jpg):
    """
    Function to generate image file paths and check their existence
    """

    # Construct the directory path
    directory_path = f'{mimic_cxr_jpg}/files/{row["patient_folder"]}/{row["patient_id"]}/{row["study_id"]}'

    # Use glob to find all .jpg files in the directory
    image_files = glob.glob(os.path.join(directory_path, '*.jpg'))

    # Print the directory and the files found
    if not image_files:
        log.info("No files found in directory: " + directory_path)

    return image_files

def resize_image(img_path:str, max_size: int=256) -> Image:
    '''
    Resize the image to have a size of max_size on the larger side
    '''
    img_copy = Image.open(img_path).copy()
    img_copy.thumbnail((max_size, max_size))
    return img_copy

def resize_and_save_images(row, base_directory, target_size=256):
    """
    Resizes and saves images from the given row of data.
    Parameters:
        row (pd.Series): A row from a DataFrame containing image file paths and metadata.
        base_directory (str): The base directory where resized images will be saved.
        target_size (int, optional): The target size for resizing the images. Default is 256.
    Returns:
        list: A list of file paths to the resized images.
    """

    save_dir = os.path.join(base_directory, row["patient_folder"], row["patient_id"], row["study_id"])
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    path = row['image_file']
    # get resized image path
    image_name = os.path.basename(path)
    resized_image_path = os.path.join(save_dir, f'{os.path.splitext(image_name)[0]}_resized.jpg')
    # resize and save image
    if not os.path.exists(resized_image_path):
        resized_image = resize_image(path, target_size)
        resized_image.save(resized_image_path)

    return resized_image_path

@hydra.main(version_base=None, config_path='../configs', config_name="preprocessing")
def main(cfg: DictConfig) -> None:

    # Constants from hydra config
    MIMIC_CXR_JPG = cfg.paths.mimic_cxr_jpg
    MIMIC_CXR_REPORTS = cfg.paths.mimic_cxr_reports
    OUTPUT_DIR = cfg.paths.output_dir
    STANZA_DIR = cfg.paths.stanza_dir
    STANZA_LANGUAGE = cfg.nlp.language
    RESIZE_SHAPE = cfg.preprocessing.resize_shape
    MIN_WORDS = cfg.preprocessing.min_sentence_words

    # Initialize pandarallel
    pandarallel.initialize(progress_bar=cfg.parallel.progress_bar)

    log.info(f'Starting preprocessing for MIMIC-CXR dataset')
    log.info(f'JPG images are located at {MIMIC_CXR_JPG}')
    log.info(f'Reports are located at {MIMIC_CXR_REPORTS}')
    log.info(f'Output directory is {OUTPUT_DIR}')

    # get patient folders
    log.info('Finding patient folders')
    patient_folders = glob.glob(f'{MIMIC_CXR_JPG}/files/p*/p*/')
    log.info(f'Found {len(patient_folders)} patient folders')

    # get study folders
    log.info('Finding study folders')
    study_ids_folders = glob.glob(f'{MIMIC_CXR_JPG}/files/p*/p*/s*/')
    log.info(f'Found {len(study_ids_folders)} study folders')

    # create a dataframe with patient_folder, patient_id, study_id
    log.info('Creating dataframe with patient_folder, patient_id, study_id')
    text_reports_df = pd.DataFrame(np.vstack(pd.Series(study_ids_folders).parallel_apply(lambda x: x.split('/')[-4:-1]).values), columns=['patient_folder', 'patient_id', 'study_id'])
    print()
    log.info(f'Created dataframe with {text_reports_df.shape[0]} rows')

    # get findings and impressions
    log.info('Extracting findings and impressions from text reports')
    text_reports_df['finding'], text_reports_df['impression'] = zip(*text_reports_df.parallel_apply(lambda x: get_findings_impressions(x, MIMIC_CXR_REPORTS), axis=1))
    print()
    log.info('Extracted findings and impressions')

    # Some reports have no findings nor impressions, we will remove them
    log.info('Removing reports with no findings nor impressions')
    valid_mask = text_reports_df.parallel_apply(
                                        lambda x: len(x['finding'])!=0 or len(x['impression'])!=0,
                                        axis=1)
    text_reports_df = text_reports_df[valid_mask]
    print()
    log.info(f'Removed {(~valid_mask).sum()} reports with no findings nor impressions')

    # Tokenize findings and impressions
    log.info('Tokenizing findings and impressions')
    if not os.path.exists(STANZA_DIR + '/' + STANZA_LANGUAGE):
        stanza.download(STANZA_LANGUAGE, model_dir=STANZA_DIR + '/' + STANZA_LANGUAGE)
    nlp = stanza.Pipeline(lang='en',
                          processors='tokenize',
                          model_dir=STANZA_DIR + '/' + STANZA_LANGUAGE)
    tqdm.pandas() # Enable progress_apply
    log.info('Tokenizing findings')
    text_reports_df['finding_sentences'] = text_reports_df['finding'].progress_apply(lambda x: get_sentences(nlp(x), min_words=MIN_WORDS))
    log.info('Tokenizing impressions')
    text_reports_df['impression_sentences'] = text_reports_df['impression'].progress_apply(lambda x: get_sentences(nlp(x), min_words=MIN_WORDS))
    # Some samples have no sentences, after tokenization, we have to remove them (around 11 samples)
    valid_mask = text_reports_df.parallel_apply(
                                        lambda x: len(x['finding_sentences'])!=0 or len(x['impression_sentences'])!=0,
                                        axis=1)
    text_reports_df = text_reports_df[valid_mask]
    log.info(f'Removed {(~valid_mask).sum()} reports with no sentences after tokenization')

    # drop raw text columns
    text_reports_df.drop(columns=['finding', 'impression'], inplace=True)

    # Find all image files
    log.info('Finding image files')
    text_reports_df['image_files'] = text_reports_df.parallel_apply(lambda x: generate_image_path(x, MIMIC_CXR_JPG),
                                                                    axis=1)
    print()

    # Explode the image files 
    log.info('Exploding image files. Each row will have one image file')
    text_reports_df = text_reports_df.explode('image_files').reset_index(drop=True)
    text_reports_df.rename(columns={'image_files': 'image_file'}, inplace=True)

    # Resize and save images
    log.info('Resizing and saving images')
    text_reports_df['resized_image_file'] = text_reports_df.parallel_apply(
                                        lambda x: resize_and_save_images(x,
                                                                        OUTPUT_DIR,
                                                                        target_size=RESIZE_SHAPE),
                                        axis=1)
    print()

    # Save the dataframe
    log.info(f'Saving the dataframe to {OUTPUT_DIR}/mimic_cxr_preprocessed.pkl')
    text_reports_df.to_pickle(f'{OUTPUT_DIR}/mimic_cxr_preprocessed.pkl')

    log.info('Preprocessing completed')

if __name__ == "__main__":
    main()