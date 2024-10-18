import os
import torch
import librosa
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd

# Dataset paths
dataset_path = "D:/cv-corpus-19.0-2024-09-13/en/clips"
test_tsv = "D:/cv-corpus-19.0-2024-09-13/en/test.tsv"

# Load the test dataframe
test_df = pd.read_csv(test_tsv, sep='\t', low_memory=False)

# Load the pretrained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cuda")  # Move model to GPU

# Function to load audio and preprocess
def load_audio(file_path):
    # Load the .mp3 file and convert it to wav-like format (in memory)
    audio = AudioSegment.from_mp3(file_path).set_frame_rate(16000)  # Set to 16kHz
    audio_samples = audio.get_array_of_samples()

    # Convert stereo to mono if necessary
    if audio.channels == 2:
        audio_samples = audio_samples.reshape((-1, 2)).mean(axis=1)  # Average channels
    
    return torch.tensor(audio_samples, dtype=torch.float32).to("cuda")  # Move audio tensor to GPU

# Select 5 random audio samples from the test set
random_samples = test_df.sample(5)

for idx, row in random_samples.iterrows():
    label = row['sentence']
    file_path = os.path.join(dataset_path, row['path'])
    
    print(f"Processing audio: {file_path}")
    print(f"Expected Label: {label}")
    
    try:
        # Load and preprocess the audio file
        audio_tensor = load_audio(file_path)
        inputs = processor(audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")  # Move inputs to GPU

        # Run the model and get predictions
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get the predicted tokens and decode them
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)
        print(f"Transcription: {transcription[0]}\n")
        
        # Export the audio for manual inspection
        audio_export = AudioSegment.from_mp3(file_path)
        audio_export.export(f"./audio_sample_{idx}.wav", format="wav")
        print(f"Exported audio to './audio_sample_{idx}.wav' for manual inspection.\n")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

