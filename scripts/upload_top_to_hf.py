#!/usr/bin/env python3
"""Upload selected top-level files from output/luoguqwencoder-lora to a Hugging Face repo.

Usage:
  python scripts/upload_top_to_hf.py --repo-id Chi-Shan/Qwen4Luogu

The script will:
- Create the repo if missing (can disable with --no-create-repo)
- Skip files that already exist on the hub with the same size
- Retry uploads on transient failures
"""

from pathlib import Path
import argparse
import os
import time

try:
    from huggingface_hub import HfApi
except Exception:
    print("Please install huggingface_hub: pip install huggingface_hub")
    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF repo id (e.g. username/Repo)")
    parser.add_argument("--local-dir", default="output/luoguqwencoder-lora", help="Local source directory")
    parser.add_argument("--path-in-repo", default="output/luoguqwencoder-lora", help="Destination path in repo")
    parser.add_argument("--no-create-repo", action="store_true", help="Don't attempt to create the repo")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff", type=int, default=2)
    parser.add_argument("--token", default=None, help="Hugging Face token (optional)")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        raise SystemExit(f"Local dir not found: {local_dir}")

    files = [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]

    api = HfApi()
    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")

    if not args.no_create_repo:
        try:
            api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, token=token)
            print(f"Ensured repo {args.repo_id} exists")
        except Exception as e:
            print("Warning: create_repo failed or repo exists:", e)

    # get remote file sizes (if available)
    remote = {}
    try:
        infos = api.list_files_info(args.repo_id)
        for info in infos or []:
            name = info.get("rfilename") or info.get("path")
            size = info.get("size")
            if name:
                # normalize: ensure path starts with the path_in_repo
                remote[name] = size
    except Exception:
        try:
            repo_files = api.list_repo_files(args.repo_id)
            for name in repo_files or []:
                remote[name] = None
        except Exception:
            print("Could not list remote files; will not skip by size.")

    summary = {"uploaded":[], "skipped":[], "failed":[]}

    for fname in files:
        local_path = local_dir / fname
        if not local_path.exists():
            print(f"Not found, skipping: {local_path}")
            continue
        dest = f"{args.path_in_repo}/{fname}"
        remote_size = remote.get(dest)
        local_size = local_path.stat().st_size
        if remote_size is not None and remote_size == local_size:
            print(f"Skipping (already present, same size): {dest}")
            summary["skipped"].append(dest)
            continue

        last_exc = None
        for attempt in range(1, args.max_retries + 1):
            try:
                print(f"Uploading ({attempt}/{args.max_retries}): {local_path} -> {dest}")
                api.upload_file(path_or_fileobj=str(local_path), path_in_repo=dest, repo_id=args.repo_id, token=token)
                print(f"Uploaded: {dest}")
                summary["uploaded"].append(dest)
                break
            except Exception as e:
                last_exc = e
                wait = args.backoff * (2 ** (attempt - 1))
                print(f"Error uploading {dest} (attempt {attempt}): {e}; retrying in {wait}s")
                time.sleep(wait)
        else:
            print(f"Failed to upload: {dest} -> {last_exc}")
            summary["failed"].append((dest, str(last_exc)))

    print("\nSummary:")
    print("Uploaded:")
    for x in summary["uploaded"]:
        print(" -", x)
    print("Skipped:")
    for x in summary["skipped"]:
        print(" -", x)
    if summary["failed"]:
        print("Failed:")
        for f, e in summary["failed"]:
            print(f" - {f}: {e}")


if __name__ == '__main__':
    main()
