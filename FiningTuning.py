import os
import pandas as pd
import re
import json
import soundfile as sf
import torch
import numpy as np
import librosa
import evaluate
from datasets import Dataset
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, 
                          Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Directories for your dataset
train_tsv = "D:/cv-corpus-19.0-2024-09-13/en/train.tsv"
test_tsv = "D:/cv-corpus-19.0-2024-09-13/en/test.tsv"
clips_dir = "D:/cv-corpus-19.0-2024-09-13/en/clips"

chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'

# Function to remove special characters and convert to lowercase
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

# Function to load the dataset from the provided tsv files
def load_data(tsv_file, num_rows=None):
    df = pd.read_csv(tsv_file, sep='\t', nrows=num_rows)
    print(f"Dataset loaded successfully from {tsv_file}!")
    return df

# Function to convert audio paths into arrays, and resample if necessary
def speech_file_to_array_fn(batch):
    try:
        audio_path = os.path.join(clips_dir, batch["path"])
        speech_array, sampling_rate = sf.read(audio_path)
        
        if sampling_rate != 16000:
            speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
        
        batch["speech"] = speech_array  # Add the speech data to the row
        batch["sampling_rate"] = 16000  # Ensure the sampling rate is set

    except Exception as e:
        print(f"Error processing audio for {batch['path']}: {e}")
        batch["speech"] = None  # Set to None if there's an issue
        batch["sampling_rate"] = None

    return batch

# Function to build vocabulary from the training data
def extract_vocabulary(train_df):
    all_text = " ".join(train_df["sentence"].values)
    vocab_list = list(set(all_text))
    
    # Add special tokens
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = len(vocab_dict)  # Map the space character
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    # Save vocabulary to a json file
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    
    print(f"Vocabulary saved to vocab.json with {len(vocab_dict)} tokens!")

# Function to prepare dataset
def prepare_dataset(batch, processor):
    batch["input_values"] = processor(batch["speech"], sampling_rate=16000, return_tensors="pt").input_values[0].numpy()
    batch["labels"] = processor(text=batch["sentence"], return_tensors="pt").input_ids[0].numpy()
    return batch

# Initialize processor
def initialize_processor():
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor

# Load the WER metric
wer_metric = evaluate.load("wer")

# Compute metrics for evaluation
def compute_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace padding token in labels (-100) with pad_token_id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False, skip_special_tokens=True)

    # Log the first few predictions and labels for manual inspection
    for i in range(5):
        print(f"Prediction {i}: {pred_str[i]}")
        print(f"Label {i}: {label_str[i]}")

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Custom Data Collator for dynamic padding
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        # Separate inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad inputs
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        # Replace padding with -100 so that padding tokens are ignored in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

# Main code execution
if __name__ == "__main__":
    # Load the train and test datasets
    train_df = load_data(train_tsv, num_rows=500)  # Reduced dataset size to fit into memory
    test_df = load_data(test_tsv, num_rows=100)    # Reduced test dataset

    # Extract and save vocabulary from training data
    extract_vocabulary(train_df)

    # Initialize processor after vocabulary is created
    processor = initialize_processor()

    # Preprocess and convert the files into audio data
    train_df = train_df.apply(speech_file_to_array_fn, axis=1)
    test_df = test_df.apply(speech_file_to_array_fn, axis=1)

    # Prepare dataset for training
    train_prepared = train_df.apply(lambda x: prepare_dataset(x, processor), axis=1)
    test_prepared = test_df.apply(lambda x: prepare_dataset(x, processor), axis=1)

    # Convert the Pandas DataFrame into a datasets.Dataset object
    train_dataset_hf = Dataset.from_pandas(train_prepared)
    test_dataset_hf = Dataset.from_pandas(test_prepared)

    # Load pretrained model
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )

    model.freeze_feature_extractor()

    # Custom Data Collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir="./wav2vec2-base-timit-demo",
        group_by_length=True,
        per_device_train_batch_size=4,  # Reduced batch size
        evaluation_strategy="steps",
        num_train_epochs=5,  # Reduced number of epochs for faster training
        fp16=True,
        save_steps=100,  # Frequent saving to avoid large memory loads
        eval_steps=100,  # Frequent evaluations
        logging_steps=100,
        learning_rate=1e-5,  # Reduced learning rate for better convergence
        weight_decay=0.005,
        warmup_steps=2000,  # Increased warmup steps
        save_total_limit=2,
        gradient_accumulation_steps=4  # Using accumulation to allow smaller batches
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        train_dataset=train_dataset_hf,
        eval_dataset=test_dataset_hf,
        data_collator=data_collator,  # Use custom data collator
        tokenizer=processor.feature_extractor  # Ensure the processor is passed
    )

    # Check and match audio-labels
    for _ in range(5):  # Display 5 random samples
        sample = test_df.sample(1).iloc[0]
        print(f"Audio file: {sample['path']}")
        print(f"Label: {sample['sentence']}\n")

    # Train the model
    trainer.train()
