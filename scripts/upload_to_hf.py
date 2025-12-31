from huggingface_hub import HfApi, login

def upload_model():
    print("--- Upload Model to Hugging Face ---")
    
    # 1. Login
    token = input("Enter your Hugging Face Write Token: ").strip()
    if not token:
        print("Token is required!")
        return
    
    login(token=token)
    
    # 2. Get Repo Details
    username = input("Enter your Hugging Face Username: ").strip()
    model_name = input("Enter a name for your new model repo (e.g., safe-or-not-bert): ").strip()
    repo_id = f"{username}/{model_name}"
    
    print(f"\nPreparing to upload 'saved_model/' to {repo_id}...")
    
    # 3. Create Repo and Upload
    api = HfApi()
    
    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True)
        print(f"Repository {repo_id} is ready.")
        
        # Upload folder
        api.upload_folder(
            folder_path="./saved_model",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial model upload from SafeOrNot project"
        )
        
        print(f"\n✅ Successfully uploaded model to: https://huggingface.co/{repo_id}")
        print("You can now use this model in your code with:")
        print(f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")')
        print(f'model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")')
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    upload_model()
