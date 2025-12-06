from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="khmin101/test3",
    repo_type="dataset"
)
print(f"다운로드된 경로: {local_path}")
