import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# Define the file path
audio_file_path = r"R:\PDS-10\harvard.wav"

# Check if the file exists
if not os.path.exists(audio_file_path):
    raise FileNotFoundError(f"Audio file not found at: {audio_file_path}")

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model ID
model_id = "openai/whisper-large-v3"

try:
    # Load model and processor
    print("Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Create the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Load and process the audio
    print("Processing audio...")
    result = pipe(audio_file_path)
    print("Transcription:")
    print(result["text"])

except ConnectionError:
    print("Network error: Unable to connect to Hugging Face servers. Please check your internet connection.")
except Exception as e:
    print(f"An error occurred: {e}")
