from huggingface_hub import HfApi
from pathlib import Path

def deploy():
    repo_id = "aryaman1222/safe"
    BASE_DIR = Path(__file__).resolve().parent.parent
    local_dir = BASE_DIR / "model" / "saved_model"
    
    print(f"Uploading {local_dir} to {repo_id}...")
    api = HfApi()
    
    try:
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ Upload successful!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    deploy()
