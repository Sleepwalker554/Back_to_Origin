import csv
from pathlib import Path
import warnings

import librosa
import numpy as np
import torch
from opensmile.core.smile import Smile
from opensmile.core.define import FeatureSet, FeatureLevel
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from config import FEAT_SEQ_LEN, SAMPLING_RATE, PROJECT_ROOT

def load_audio(file_path: str) -> np.ndarray:
    """ 
    Args:
        file_path: Audio file path
    
    Returns:
        audio_array: numpy array, mono audio
    """
    # Use librosa to load (supports MP3 and WAV)
    array, _ = librosa.load(file_path, sr=SAMPLING_RATE, res_type="kaiser_best")
    # Ensure mono
    array = librosa.to_mono(array)
    # Convert to float32
    array = np.float32(array)
    return array

class CsvDataset(Dataset):
    """ 
    Load session_id and audio path from CSV
    """
    def __init__(self, csv_path: Path, raw_audio_dir: Path):
        super().__init__()
        
        self.csv_path = csv_path
        self.raw_audio_dir = raw_audio_dir
        self.data = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row['session_id']
                egemaps_path = row['egemaps_path']
                
                # Build audio path
                # From CSV's third column (ad) to determine audio path in Control or Dementia folder
                ad = int(row['ad'])
                if ad == 0:
                    folder = "Control"
                else:
                    folder = "Dementia"
                
                # Try to find audio file in .wav or .mp3 format
                audio_path_wav = self.raw_audio_dir / folder / f"{session_id}.wav"
                audio_path_mp3 = self.raw_audio_dir / folder / f"{session_id}.mp3"
                
                # Determine which audio file exists
                if audio_path_wav.exists():
                    audio_path = audio_path_wav
                elif audio_path_mp3.exists():
                    audio_path = audio_path_mp3
                else:
                    # Default to .wav (will be caught as not existing later)
                    audio_path = audio_path_wav
                
                self.data.append({
                    'session_id': session_id,
                    'audio_path': str(audio_path.relative_to(PROJECT_ROOT)),
                    'egemaps_path': egemaps_path,
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item['audio_path'], item['egemaps_path'], item['session_id']


def extract_egemaps_features_from_csv(csv_path: Path, raw_audio_dir: Path):
    """
    Args:
        csv_path: CSV file path
        raw_audio_dir: Original audio directory (contains Control and Dementia subfolders)
    """
    # Create dataset
    dataset = CsvDataset(csv_path, raw_audio_dir=raw_audio_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Process one by one
        shuffle=False,
        num_workers=0,     # OpenSMILE does not support multiple processes
        persistent_workers=False,
    )
    
    print(f"\n============= Extraction eGeMaps features =============")
    print(f"{len(dataset)} Audio Files")

    # Initialize OpenSMILE
    smile_lld = Smile(
        feature_set=FeatureSet.eGeMAPSv02,
        feature_level=FeatureLevel.LowLevelDescriptors,
    )
    
    # Count the number of extracted features
    extracted = 0
    skipped = 0
    error = 0
    
    # Ignore OpenSMILE warnings
    warnings.simplefilter('ignore')
    
    # Extract features
    for audio_path, egemaps_path, session_id in tqdm(dataloader, desc="Extracting"):
        
        # Convert to absolute path
        audio_path_abs = PROJECT_ROOT / audio_path
        egemaps_path_abs = PROJECT_ROOT / egemaps_path
        
        # Check if it already exists
        if egemaps_path_abs.exists():
            skipped += 1
            continue
        
        # Check if the audio file exists
        if not audio_path_abs.exists():
            print(f"\nWarning: Audio file does not exist: {audio_path_abs}")
            error += 1
            continue
        
        try:
            # Load audio
            audio_np = load_audio(str(audio_path_abs))
            
            # Audio segmentation (remove the part that is not enough for one segment)
            usable_length = (audio_np.shape[0] // FEAT_SEQ_LEN) * FEAT_SEQ_LEN
            
            if usable_length == 0:
                print(f"\nAudio too short, skipping extraction: {session_id}")
                continue
            
            # Extract and segment
            audio_segments = np.split(audio_np[:usable_length], FEAT_SEQ_LEN)
            
            # Extract eGeMAPS features segment by segment
            egemaps_list = []
            for segment in audio_segments:
                # OpenSMILE processing
                _, _, features = smile_lld.process(segment, SAMPLING_RATE)
                # features shape: (1, 25) - Take the first frame
                feat_np = np.array(features[0, :], dtype=np.float32)
                egemaps_list.append(torch.from_numpy(feat_np))
            
            # (FEAT_SEQ_LEN, 25) Tensor
            egemaps = torch.stack(egemaps_list, dim=0)
 
            egemaps_path_abs.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(egemaps, egemaps_path_abs)
            
            extracted += 1
            
        except Exception as e:
            print(f"\nError: Extraction failed {session_id}: {e}")
            error += 1
            continue
 
    print(f"Successfully extracted: {extracted}")
    print(f"Already Exists (Skipped): {skipped}")
    print(f"Total: {len(dataset)}")
    print(f"Errors: {error}")