import os
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import evaluate

# Step 1: Check if CUDA is available for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Define the path to your dataset
dataset_path = "D:/cv-corpus-19.0-2024-09-13/en/clips"
train_tsv = "D:/cv-corpus-19.0-2024-09-13/en/train.tsv"
test_tsv = "D:/cv-corpus-19.0-2024-09-13/en/test.tsv"

# Load matched labels for train and test datasets with low_memory=False to avoid warnings
train_df = pd.read_csv(train_tsv, sep='\t', low_memory=False)
test_df = pd.read_csv(test_tsv, sep='\t', low_memory=False)

# Use a subset for now: 1000 from train and 100 from test
train_df = train_df[:1000]
test_df = test_df[:100]

# Step 3: On-the-Fly Conversion function to handle mp3 to wav
def load_and_convert_clip(file_path):
    try:
        # Load the .mp3 file and convert it to wav-like format (in memory)
        audio = AudioSegment.from_mp3(file_path).set_frame_rate(16000)  # Set to 16kHz
        audio_samples = audio.get_array_of_samples()

        # Convert stereo to mono if necessary
        if audio.channels == 2:
            audio_samples = audio_samples.reshape((-1, 2)).mean(axis=1)  # Average channels
        
        return torch.tensor(audio_samples, dtype=torch.float32)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to load audio files and prepare for training
def load_audio_files(df, clips_dir):
    data = []
    for idx, row in df.iterrows():
        file_path = os.path.join(clips_dir, row['path'])
        audio = load_and_convert_clip(file_path)
        if audio is not None:
            data.append({"audio": audio, "text": row['sentence']})
    return data

# Load train and test audio using on-the-fly conversion
train_data = load_audio_files(train_df, dataset_path)
test_data = load_audio_files(test_df, dataset_path)

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_dict({"audio": [d["audio"] for d in train_data], "text": [d["text"] for d in train_data]})
test_dataset = Dataset.from_dict({"audio": [d["audio"] for d in test_data], "text": [d["text"] for d in test_data]})

# Step 4: Preprocess function to prepare the data for Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

def preprocess(batch):
    batch["input_values"] = processor(batch["audio"], sampling_rate=16000, return_tensors="pt", padding=True).input_values
    batch["labels"] = processor.tokenizer(batch["text"], return_tensors="pt", padding=True).input_ids
    return batch

train_dataset = train_dataset.map(preprocess, remove_columns=["audio", "text"], batched=True)
test_dataset = test_dataset.map(preprocess, remove_columns=["audio", "text"], batched=True)

# Step 5: Load the Model from the Checkpoint
checkpoint_dir = "./results/checkpoint-250"  # Path to the saved checkpoint
model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir).to(device)

# Step 6: Data Collator and Trainer Setup
def data_collator(features):
    input_values = pad_sequence([torch.tensor(f["input_values"]) for f in features], batch_first=True, padding_value=0.0)
    labels = pad_sequence([torch.tensor(f["labels"]) for f in features], batch_first=True, padding_value=-100)
    return {"input_values": input_values, "labels": labels}

# Step 7: Define training arguments (used for evaluation, not training)
training_args = TrainingArguments(
    output_dir="./results",          
    per_device_eval_batch_size=4,    
    logging_dir="./logs",            
)

# Step 8: Initialize the WER Metric
wer_metric = evaluate.load("wer")

# Postprocess function for computing WER
def compute_metrics(pred):
    # Convert pred_logits (numpy) to PyTorch tensors
    pred_logits = torch.tensor(pred.predictions)
    
    # Get predicted labels using argmax
    pred_ids = torch.argmax(pred_logits, dim=-1)

    # Decode predicted and true labels into strings
    pred_str = processor.batch_decode(pred_ids)
    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, group_tokens=False)

    # Debug: Print predictions and labels to inspect
    print("Predictions:", pred_str[:5])  # Print first 5 predictions for inspection
    print("Labels:", label_str[:5])      # Print first 5 true labels for inspection

    # Compute the Word Error Rate (WER)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}



# Step 9: Initialize Trainer for Evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Include WER computation
)

# Step 10: Evaluate the model
results = trainer.evaluate()

# Output evaluation results
print(f"Evaluation results: {results}")
