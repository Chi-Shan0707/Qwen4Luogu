#!/usr/bin/env python3
"""Upload a local folder to a Hugging Face Hub repo folder while preserving structure.

Usage examples:
  # Dry run (just list files):
  python scripts/upload_to_hf.py --repo-id username/Qwen4Luogu --dry-run

  # Upload entire local `output` -> repo `output` (creates repo if needed):
  python scripts/upload_to_hf.py --repo-id username/Qwen4Luogu --local-dir output --path-in-repo output --private

Notes:
- The script uses `huggingface_hub`. If you already ran `huggingface-cli login`, the cached token will be used automatically.
- For a large number of files, increase `--max-workers`.
"""

from pathlib import Path
import argparse
import os
import sys

try:
    from huggingface_hub import HfApi
except Exception as e:
    print("Error: huggingface_hub is required. Install with `pip install huggingface_hub`.")
    raise


def upload_folder(api: HfApi, local_dir: Path, repo_id: str, path_in_repo: str = "output", repo_type: str = "model", max_workers: int = 1, token: str | None = None, skip_existing: bool = True, verify_size: bool = True, max_retries: int = 5, backoff_factor: int = 2, force: bool = False):
    if not local_dir.exists():
        raise SystemExit(f"Local directory not found: {local_dir}")

    # Try the convenient API first (only when multi-file parallel upload is acceptable)
    try:
        print(f"Trying HfApi.upload_folder: {local_dir} -> {repo_id}/{path_in_repo}")
        try:
            api.upload_folder(folder_path=str(local_dir), path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type, token=token, max_workers=max_workers)
        except TypeError:
            api.upload_folder(local_dir=str(local_dir), path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type, token=token, max_workers=max_workers)
        print("upload_folder finished.")
        return
    except Exception as e:
        print("upload_folder not available or failed, falling back to per-file uploads:", e)

    # Build list of files to upload
    files = [p for p in sorted(local_dir.rglob("*")) if p.is_file()]
    if not files:
        print("No files to upload.")
        return

    # Get remote file info (try to obtain sizes)
    remote = {}
    try:
        # list_files_info gives size information in newer versions
        infos = api.list_files_info(repo_id)
        for info in infos:
            try:
                # FileInfo objects may have 'rfilename' or 'path' and 'size'
                name = getattr(info, "rfilename", None) or getattr(info, "path", None) or info.get("rfilename")
                size = getattr(info, "size", None) or info.get("size")
                if name:
                    remote[name] = size
            except Exception:
                continue
    except Exception:
        # Fallback: names only
        try:
            repo_files = api.list_repo_files(repo_id)
            for name in repo_files or []:
                remote[name] = None
        except Exception:
            print("Could not list repo files (continuing without skip/size checks).")

    import time

    uploaded = 0
    skipped = 0
    skipped_size_mismatch = 0
    failed = 0
    failures = []

    def should_skip(dest: str, local_path: Path):
        if force:
            return False, "force"
        if dest not in remote:
            return False, "not_remote"
        # remote has entry; if verify_size and remote size available, compare
        remote_size = remote.get(dest)
        local_size = local_path.stat().st_size
        if verify_size and remote_size is not None:
            if remote_size == local_size:
                return True, "same_size"
            else:
                return False, "size_mismatch"
        # if verify_size requested but remote size unknown, don't skip
        if verify_size and remote_size is None:
            return False, "unknown_size"
        # if not verifying size, skip if name exists
        return True, "exists"

    def attempt_upload(file_path: Path):
        nonlocal uploaded, skipped, skipped_size_mismatch, failed
        rel = file_path.relative_to(local_dir)
        dest = f"{path_in_repo}/{rel.as_posix()}"

        skip, reason = should_skip(dest, file_path) if skip_existing else (False, "skip_disabled")
        if skip:
            if reason == "same_size" or reason == "exists":
                print(f"Skipping existing (matched): {dest}")
                skipped += 1
                return True
            else:
                # size mismatch - will reupload
                print(f"Remote exists but differs (will reupload): {dest}")
                skipped_size_mismatch += 1

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Uploading ({attempt}/{max_retries}): {file_path} -> {dest}")
                api.upload_file(path_or_fileobj=str(file_path), path_in_repo=dest, repo_id=repo_id, repo_type=repo_type, token=token)
                print(f"Uploaded: {dest}")
                uploaded += 1
                return True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                last_exc = e
                wait = backoff_factor * (2 ** (attempt - 1))
                print(f"Error uploading {dest} (attempt {attempt}/{max_retries}): {e}; retrying in {wait}s")
                time.sleep(wait)
        failed += 1
        failures.append((dest, last_exc))
        return False

    print(f"Starting per-file uploads (single-thread friendly) with max_workers={max_workers}...")
    try:
        if max_workers == 1:
            # simple sequential loop (better when network unstable)
            for f in files:
                attempt_upload(f)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(attempt_upload, p): p for p in files}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        p = futures[fut]
                        print(f"Unhandled exception for {p}: {e}")
                        failed += 1
                        failures.append((str(p), e))
    except KeyboardInterrupt:
        print("Upload interrupted by user (KeyboardInterrupt).")

    print("Upload summary:")
    print(f"  Total files considered: {len(files)}")
    print(f"  Uploaded: {uploaded}")
    print(f"  Skipped (name match): {skipped}")
    print(f"  Skipped due to size match: {skipped_size_mismatch}")
    print(f"  Failed: {failed}")
    if failures:
        print("Failures details:")
        for dest, exc in failures:
            print(f" - {dest}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Upload a local folder to Hugging Face Hub repo folder (preserve structure)")
    parser.add_argument("--repo-id", default="Qwen4Luogu", help="Repo id on HF (e.g. username/Qwen4Luogu or Qwen4Luogu)")
    parser.add_argument("--local-dir", default="output", help="Local directory to upload (default: output)")
    parser.add_argument("--path-in-repo", default="output", help="Destination path inside repo (default: output)")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"], help="Repo type (default: model)")
    parser.add_argument("--private", action="store_true", help="Create repo as private if creating it now")
    parser.add_argument("--max-workers", type=int, default=1, help="Max parallel uploads (default: 1; set >1 only if your network is stable)")
    parser.add_argument("--verify-size/--no-verify-size", dest="verify_size", default=True, help="Verify remote file size and skip upload when sizes match (default: on)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars (useful when running in unstable terminals)")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per file (default: 5)")
    parser.add_argument("--backoff-factor", type=int, default=2, help="Backoff factor in seconds for retries (default: 2)")
    parser.add_argument("--force", action="store_true", help="Force re-upload even if file exists on remote")
    parser.add_argument("--token", default=None, help="Hugging Face token (optional). If omitted, cached token or env var will be used.")
    parser.add_argument("--create-repo/--no-create-repo", dest="create_repo", default=True, help="Create repo if it doesn't exist (default: True)")
    parser.add_argument("--dry-run", action="store_true", help="Just list files that would be uploaded")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    api = HfApi()

    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")

    if args.create_repo:
        try:
            print(f"Ensuring repo {args.repo_id} exists (private={args.private})...")
            api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True, token=token)
        except Exception as e:
            print("Warning: create_repo failed or repo already exists:", e)

    if args.dry_run:
        if not local_dir.exists():
            raise SystemExit(f"Local directory not found: {local_dir}")
        print("Dry run: the following files would be uploaded:")
        for p in sorted(local_dir.rglob("*")):
            if p.is_file():
                print(p)
        return

    # Optionally disable progress bars from huggingface_hub
    if args.no_progress:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    upload_folder(api, local_dir, args.repo_id, args.path_in_repo, args.repo_type, args.max_workers, token)

    local_dir = Path(args.local_dir)
    api = HfApi()

    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")

    if args.create_repo:
        try:
            print(f"Ensuring repo {args.repo_id} exists (private={args.private})...")
            api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True, token=token)
        except Exception as e:
            print("Warning: create_repo failed or repo already exists:", e)

    if args.dry_run:
        if not local_dir.exists():
            raise SystemExit(f"Local directory not found: {local_dir}")
        print("Dry run: the following files would be uploaded:")
        for p in sorted(local_dir.rglob("*")):
            if p.is_file():
                print(p)
        return

    upload_folder(api, local_dir, args.repo_id, args.path_in_repo, args.repo_type, args.max_workers, token)


if __name__ == "__main__":
    main()
