import os
import pandas as pd
import torch
import librosa
from pydub import AudioSegment
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments

# Step 1: Check if CUDA is available for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Define paths to your train/test dataset and clips directory
train_tsv = "D:/cv-corpus-19.0-2024-09-13/en/train.tsv"
test_tsv = "D:/cv-corpus-19.0-2024-09-13/en/test.tsv"
clips_dir = "D:/cv-corpus-19.0-2024-09-13/en/clips"
wav_clips_dir = "E:/Whioo/Sem VII/Sem 7 Current/Minor Project/Lets start/wav_clips"  # Directory to store converted .wav files

# Ensure the converted wav_clips directory exists
os.makedirs(wav_clips_dir, exist_ok=True)

# Step 3: Load the train and test datasets
train_df = pd.read_csv(train_tsv, sep='\t')
test_df = pd.read_csv(test_tsv, sep='\t')

# Step 4: Convert .mp3 files to .wav if not already converted
def convert_mp3_to_wav(mp3_file, wav_dir):
    wav_file = os.path.join(wav_dir, os.path.basename(mp3_file).replace(".mp3", ".wav"))
    if not os.path.exists(wav_file):
        try:
            audio = AudioSegment.from_mp3(mp3_file)
            audio = audio.set_frame_rate(16000)  # Set sampling rate to 16kHz
            audio.export(wav_file, format="wav")
            print(f"Converted {mp3_file} to {wav_file}")
        except Exception as e:
            print(f"Error converting {mp3_file}: {e}")
            return None
    return wav_file

# Convert files listed in train/test datasets and keep track of matched .wav files
def load_and_convert_clips(df, clips_dir, wav_clips_dir):
    converted_files = []
    for _, row in df.iterrows():
        mp3_file = os.path.join(clips_dir, row['path'])
        wav_file = convert_mp3_to_wav(mp3_file, wav_clips_dir)
        if wav_file:
            converted_files.append({"wav_file": wav_file, "sentence": row["sentence"]})
    return converted_files

print("Loading and converting train dataset...")
train_files = load_and_convert_clips(train_df, clips_dir, wav_clips_dir)
print(f"Converted {len(train_files)} train files.")

print("Loading and converting test dataset...")
test_files = load_and_convert_clips(test_df, clips_dir, wav_clips_dir)
print(f"Converted {len(test_files)} test files.")

# Step 5: Load the Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)

# Step 6: Preprocess the audio files
def preprocess_function(examples):
    # Load the audio
    audio, _ = librosa.load(examples['wav_file'], sr=16000)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    return {"input_values": input_values[0].numpy(), "labels": processor.tokenizer(examples['sentence']).input_ids}

train_dataset = Dataset.from_dict({"wav_file": [f["wav_file"] for f in train_files], "sentence": [f["sentence"] for f in train_files]})
test_dataset = Dataset.from_dict({"wav_file": [f["wav_file"] for f in test_files], "sentence": [f["sentence"] for f in test_files]})

train_dataset = train_dataset.map(preprocess_function, remove_columns=["wav_file", "sentence"], batched=True)
test_dataset = test_dataset.map(preprocess_function, remove_columns=["wav_file", "sentence"], batched=True)

# Step 7: Define the data collator
def data_collator(features):
    input_values = [torch.tensor(feature["input_values"]) for feature in features]
    labels = [torch.tensor(feature["labels"]) for feature in features]
    batch = {"input_values": torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)}
    batch["labels"] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return batch

# Step 8: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2
)

# Step 9: Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor
)

# Step 10: Train the model
print("Starting training...")
trainer.train()

# Step 11: Save the fine-tuned model and processor
model.save_pretrained("./fine_tuned_wav2vec2")
processor.save_pretrained("./fine_tuned_wav2vec2")

print("Training complete. Model saved.")
