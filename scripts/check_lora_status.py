import subprocess
import os

def run_command(cmd):
    """Run a shell command and return the result."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def get_remote_branch():
    """Get the default remote branch name."""
    result = run_command('git branch -r')
    if result.returncode != 0:
        return None
    branches = result.stdout.strip().split('\n')
    for branch in branches:
        branch = branch.strip()
        if 'origin/HEAD' in branch:
            # origin/HEAD -> origin/main
            return branch.split(' -> ')[1]
        elif branch.startswith('origin/') and 'HEAD' not in branch:
            return branch
    return 'origin/main'  # default assumption

def check_lora_status():
    """Check the status of LoRA files."""
    lora_dir = 'output/luoguqwencoder-lora/'

    # Check if it's a Git repository
    if not os.path.exists('.git'):
        print("当前目录不是Git仓库。")
        return

    # Check if LoRA directory exists
    if not os.path.exists(lora_dir):
        print("LoRA目录不存在。")
        return

    print("检查LoRA文件状态...")

    # Get git status for LoRA files
    result = run_command(f'git status --porcelain {lora_dir}')
    if result.returncode != 0:
        print("运行git status失败。")
        return

    status_output = result.stdout.strip()
    if status_output:
        print("LoRA文件在Git中的状态：")
        print(status_output)
    else:
        print("LoRA文件已提交或未修改。")

    # Check if files are tracked
    result = run_command(f'git ls-files {lora_dir}')
    tracked_files = result.stdout.strip().split('\n') if result.returncode == 0 else []
    tracked_files = [f for f in tracked_files if f.strip()]

    if tracked_files:
        print(f"已跟踪的LoRA文件：{len(tracked_files)} 个")
        for f in tracked_files:
            print(f"  - {f}")
    else:
        print("没有LoRA文件被Git跟踪。")

    # Check if pushed to remote
    remote_branch = get_remote_branch()
    if remote_branch:
        result = run_command(f'git log --oneline {remote_branch}..HEAD -- {lora_dir}')
        if result.returncode == 0:
            local_commits = result.stdout.strip().split('\n')
            local_commits = [c for c in local_commits if c.strip()]
            if local_commits:
                print(f"有 {len(local_commits)} 个本地提交包含LoRA文件未推送到GitHub：")
                for commit in local_commits:
                    print(f"  - {commit}")
            else:
                print("所有包含LoRA文件的本地提交都已推送到GitHub。")
        else:
            print("检查推送状态失败。")
    else:
        print("未找到远程分支，无法检查推送状态。")

if __name__ == "__main__":
    check_lora_status()