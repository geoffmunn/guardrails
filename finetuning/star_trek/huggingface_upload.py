from huggingface_hub import HfApi
from transformers import AutoTokenizer
import os

# Ensure your training script has finished and saved the model to OUTPUT_DIR
model_directory = "./finetuned" # Use your OUTPUT_DIR variable

api = HfApi()

# Create the repository first
username = input("Enter your Hugging Face username: ").strip()
repo_name = input("Enter the repository name: ").strip()
repo_id = f"{username}/{repo_name}"

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=True, # Set to True if you want a private repo
    exist_ok=True, # This prevents an error if the repo already exists
)

# Now upload the model directory
api.upload_folder(
    folder_path=model_directory,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload fine-tuned classification model",
)

print(f"Model uploaded successfully to https://huggingface.co/{repo_id}")