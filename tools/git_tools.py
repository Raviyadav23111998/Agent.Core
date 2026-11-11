"""
Agent — Git Action Tools
=====================================
Git operations for code-fixing agents.

Tools:
- git_branch:  Create and checkout a new branch
- git_commit:  Stage all changes and commit
- git_status:  Show repo status
- git_diff:    Show current changes

All operations use subprocess to call git directly — no dependencies.
"""

import subprocess
import logging

logger = logging.getLogger(__name__)


def _run_git(repo_path: str, *args, timeout: int = 30) -> tuple:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip() or output
            return False, error
        return True, output
    except subprocess.TimeoutExpired:
        return False, f"Git command timed out after {timeout}s"
    except FileNotFoundError:
        return False, "git not found in PATH"
    except Exception as e:
        return False, str(e)


class GitActionTools:
    """Git operations for code agents."""

    def register_tools(self, mcp_server):

        @mcp_server.tool()
        def git_branch(repo_path: str, branch_name: str) -> str:
            """Create and checkout a new git branch.

            Creates branch from current HEAD and switches to it.

            Args:
                repo_path: Absolute path to the git repository
                branch_name: Branch name (e.g., "fix/SUP-42", "feature/auth-refactor")
            """
            ok, output = _run_git(repo_path, "branch", "--list", branch_name)
            if ok and output.strip():
                ok, output = _run_git(repo_path, "checkout", branch_name)
                if ok:
                    return f"✓ Switched to existing branch '{branch_name}'"
                return f"✗ Checkout failed: {output}"

            ok, output = _run_git(repo_path, "checkout", "-b", branch_name)
            if ok:
                return f"✓ Created and switched to branch '{branch_name}'"
            return f"✗ Branch creation failed: {output}"

        @mcp_server.tool()
        def git_commit(repo_path: str, message: str) -> str:
            """Stage all changes and create a commit.

            Stages all modified/new files, then commits with the given message.

            Args:
                repo_path: Absolute path to the git repository
                message: Commit message
            """
            ok, output = _run_git(repo_path, "add", "-u")
            if not ok:
                return f"✗ Stage failed: {output}"

            ok, status = _run_git(repo_path, "diff", "--cached", "--stat")
            if ok and not status.strip():
                return "⚠ Nothing to commit (no staged changes)."

            ok, output = _run_git(repo_path, "commit", "-m", message)
            if ok:
                _, hash_out = _run_git(repo_path, "rev-parse", "--short", "HEAD")
                return f"✓ Committed [{hash_out}]: {message}"
            return f"✗ Commit failed: {output}"

        @mcp_server.tool()
        def git_status(repo_path: str) -> str:
            """Show git repository status.

            Args:
                repo_path: Absolute path to the git repository
            """
            _, branch = _run_git(repo_path, "branch", "--show-current")

            ok, status = _run_git(repo_path, "status", "--short")
            if not ok:
                return f"✗ Status failed: {status}"

            _, log = _run_git(repo_path, "log", "--oneline", "-5")

            lines = [f"Branch: {branch}"]
            if status:
                lines.append(f"\nChanges:\n{status}")
            else:
                lines.append("\nWorking tree clean.")
            if log:
                lines.append(f"\nRecent commits:\n{log}")

            return "\n".join(lines)

        @mcp_server.tool()
        def git_diff(repo_path: str, staged: str = "false") -> str:
            """Show current changes (diff).

            Args:
                repo_path: Absolute path to the git repository
                staged: "true" to show staged changes, "false" for unstaged
            """
            args = ["diff"]
            if staged.lower() == "true":
                args.append("--cached")
            args.append("--stat")

            ok, stat = _run_git(repo_path, *args)
            if not ok:
                return f"✗ Diff failed: {stat}"

            if not stat.strip():
                return "No changes."

            diff_args = ["diff"]
            if staged.lower() == "true":
                diff_args.append("--cached")

            _, full_diff = _run_git(repo_path, *diff_args)
            if len(full_diff) > 3000:
                full_diff = full_diff[:3000] + f"\n\n... (truncated, {len(full_diff)} chars total)"

            return f"Stats:\n{stat}\n\nDiff:\n{full_diff}"
