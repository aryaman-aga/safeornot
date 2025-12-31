from huggingface_hub import HfApi

def deploy():
    repo_id = "aryaman1222/safe"
    local_dir = "./saved_model"
    
    print(f"Uploading {local_dir} to {repo_id}...")
    api = HfApi()
    
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ Upload successful!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    deploy()
