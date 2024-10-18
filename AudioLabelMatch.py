import os
import librosa
import pandas as pd
from pydub import AudioSegment

# Function to check 5 random samples from the dataset and load their corresponding clips
def check_random_samples(df, clips_dir, num_samples=5):
    samples = df.sample(num_samples)
    for idx, row in samples.iterrows():
        label = row['sentence']
        audio_path = os.path.join(clips_dir, row['path'])
        
        print(f"Label {idx}: {label}")
        print(f"Audio Path: {audio_path}")
        
        try:
            # Load the audio using librosa and print its shape (to ensure it's loaded correctly)
            audio, sr = librosa.load(audio_path, sr=16000)
            print(f"Audio Loaded Successfully - Sample Rate: {sr}, Audio Shape: {audio.shape}")
            
            # Export the audio to wav for manual inspection if needed
            audio_segment = AudioSegment.from_mp3(audio_path)
            audio_segment.export(f"./sample_{idx}.wav", format="wav")
            print(f"Exported audio to sample_{idx}.wav for manual inspection.\n")
        except Exception as e:
            print(f"Error loading or processing {audio_path}: {e}\n")

# Check 5 random samples from the train and test sets
def check_labels_and_clips(train_df, test_df, dataset_path, num_samples=5):
    print("Checking Random Train Samples:")
    check_random_samples(train_df, dataset_path, num_samples)

    print("Checking Random Test Samples:")
    check_random_samples(test_df, dataset_path, num_samples)

# Dataset paths
dataset_path = "D:/cv-corpus-19.0-2024-09-13/en/clips"
train_tsv = "D:/cv-corpus-19.0-2024-09-13/en/train.tsv"
test_tsv = "D:/cv-corpus-19.0-2024-09-13/en/test.tsv"

# Load the train and test dataframes
train_df = pd.read_csv(train_tsv, sep='\t', low_memory=False)
test_df = pd.read_csv(test_tsv, sep='\t', low_memory=False)

# Call the function to check labels and audio clips
check_labels_and_clips(train_df, test_df, dataset_path, num_samples=5)
