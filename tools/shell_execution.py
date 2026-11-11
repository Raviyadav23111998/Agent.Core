import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

_FORBIDDEN_PATTERN = re.compile(
    r'(?:^|[\s|;&])(nano|vim|vi|emacs|less|more|htop|top)(?:\s|$)'
)


class ShellExecutionTool:
    """Shell command execution with optional sandbox enforcement.

    Security model:
    - sandboxed=False (CLI/local): Unrestricted (OS user's permissions).
    - sandboxed=True  (Cloud/Web): Commands JAILED to workspace_root via a
      kernel-enforced bubblewrap jail built on an ALLOWLIST:

        * Filesystem: ONLY the OS runtime dirs (/usr, /bin, /lib…) are bound
          read-only, plus a handful of non-secret /etc files. Everything else
          on the host — the agent `.env`, /var/www source, /opt, other
          tenants' workspaces, the host user list — is simply ABSENT. A newly
          added secret path is invisible by default. (The previous model bound
          the whole `/` read-only and then masked a few paths; any unmasked
          secret leaked. CVE-class fix 2026-06-13.)
        * Identity: synthetic /etc/passwd + /etc/group (no host users).
        * Environment: CLEARED — the agent process env carries DB creds + API
          keys (systemd EnvironmentFile=.env); only PATH/HOME/locale pass
          through, so `printenv` can't dump secrets.
        * Network: unshared by default (`shell.sandbox_network=off`) so the
          jail cannot reach localhost Redis/Postgres/Elasticsearch/Neo4j/the
          internal API, nor exfiltrate to the internet. Opt-in per deployment.
        * Namespaces: PID/IPC/UTS/cgroup unshared; private /tmp, /run, /dev.
      * The regex layers below remain as defense-in-depth and are the ONLY
        layer on hosts without bwrap (macOS dev) — logged once as a warning.
      * cwd must be within workspace (realpath checked).

    Architecture:
    - Fast commands: run synchronously with timeout, return output
    - Slow commands: exceed timeout → auto-spawn as background task
    - Output capped at MAX_OUTPUT_BYTES to prevent context blowup
    """

    MAX_OUTPUT_BYTES = 50_000

    _BWRAP_PATH = shutil.which("bwrap") if sys.platform.startswith("linux") else None
    _os_sandbox_warned = False

    _RO_ALLOWLIST = (
        "/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32", "/libx32",
    )
    _ETC_ALLOWLIST = (
        "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d",
        "/etc/alternatives", "/etc/localtime", "/etc/mime.types",
        "/etc/ssl", "/etc/pki", "/etc/ca-certificates",
        "/etc/ca-certificates.conf",
        "/etc/resolv.conf", "/etc/nsswitch.conf", "/etc/hosts",
        "/etc/host.conf", "/etc/gai.conf",
    )
    _ENV_PASSTHROUGH = ("LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ")

    _synth_etc_dir = None
    
    _SANDBOX_BLOCKED = re.compile(
        r'(?:^|[\s|;&])'
        r'(ssh|scp|rsync|mount|umount|chroot|docker|kubectl|'
        r'systemctl|service|journalctl|'
        r'useradd|userdel|usermod|passwd|sudo|su|'
        r'iptables|ufw|ncat|socat|nc|'
        r'chown|chmod\s+[0-7]*[sS])'
        r'(?:\s|$)'
    )
    
    _SYMLINK_PATTERN = re.compile(r'ln\s+.*-s')
    
    def __init__(self, workspace_root: str, background_manager=None, sandboxed: bool = False):
        self.workspace_root = os.path.realpath(os.path.abspath(workspace_root))
        self.background_manager = background_manager
        self.sandboxed = sandboxed
        # Working directory persists across run_command calls (Claude Code's
        # persistent-shell behavior): a `cd` in one command carries to the
        # next, so multi-step jobs don't re-pay setup on every call.
        self._session_cwd = self.workspace_root
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def recommended_policy(self):
        """The ToolPolicy this shell instance warrants — decided by the OWNER
        of the sandbox state, not guessed from the tool name.

        Kernel-jailed (bwrap: no network, no host filesystem, cleared env)
        → side_effect 'write': no HITL approval, no double-send dedup — a
        jailed command has no outward blast radius, and approval-gating it
        was stalling real coding turns for up to 300s per command (Claude
        Code likewise auto-allows sandboxed bash).

        Unsandboxed, or sandboxed WITHOUT bwrap (regex-only fallback on
        non-Linux hosts) → 'irreversible': the approval gate stays.
        """
        from tools.tool_policy import ToolPolicy
        kernel_jailed = (self.sandboxed
                         and self._BWRAP_PATH is not None
                         and self._os_sandbox_mode() != "off")
        if kernel_jailed:
            return ToolPolicy(side_effect="write", idempotent=False,
                              cacheable=False, timeout_s=180.0, retries=0)
        return ToolPolicy(side_effect="irreversible", idempotent=False,
                          cacheable=False, timeout_s=180.0, retries=0)

    _CWD_MARKER = "__AG_CWD__"
    _CWD_MARKER_RE = re.compile(r"\n?__AG_CWD__(.*?)__AG_END__\s*$")

    def _wrap_for_cwd_tracking(self, command_line: str) -> str:
        """Append a cwd marker so a `cd` inside the command persists."""
        return (f"{command_line}\n"
                f"__AG_EC=$?; printf '\\n{self._CWD_MARKER}%s__AG_END__' \"$PWD\"; "
                f"exit $__AG_EC")

    def _consume_cwd_marker(self, stdout: str) -> str:
        """Strip the marker from output and persist the reported cwd."""
        m = self._CWD_MARKER_RE.search(stdout or "")
        if not m:
            return stdout
        new_cwd = (m.group(1) or "").strip()
        stdout = stdout[:m.start()]
        if new_cwd and os.path.isdir(new_cwd):
            real = os.path.realpath(new_cwd)
            if (not self.sandboxed
                    or real == self.workspace_root
                    or real.startswith(self.workspace_root + os.sep)):
                self._session_cwd = real
        return stdout
    
    RLIMIT_CPU_S = 120
    RLIMIT_FSIZE_MB = 256
    RLIMIT_NOFILE = 256

    def _sandbox_preexec(self):
        """Build a preexec_fn that applies kernel resource limits + setsid.

        Returns None on platforms without the resource module (Windows) —
        the command still runs, just without rlimits.
        """
        if not self.sandboxed:
            return None
        try:
            import resource
        except ImportError:
            return None
        cpu_s = self.RLIMIT_CPU_S
        fsize = self.RLIMIT_FSIZE_MB * 1024 * 1024
        nofile = self.RLIMIT_NOFILE

        def _limits():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
            except (ValueError, OSError):
                pass
        return _limits

    def _os_sandbox_mode(self) -> str:
        """Config `shell.os_sandbox`: auto (default) | off | require."""
        try:
            from core.config import app_config
            return (app_config.get("shell.os_sandbox", "auto") or "auto").strip().lower()
        except Exception:
            return "auto"

    def _sandbox_network_allowed(self) -> bool:
        """Config `shell.sandbox_network`: off (default, isolated) | on.

        Default OFF: the jail gets no network namespace, so it cannot reach
        localhost services (Redis/Postgres/ES/Neo4j/internal API) or the
        internet. Turn on only for deployments that need outbound network in
        the agent's shell (and have other egress controls).
        """
        try:
            from core.config import app_config
            return app_config.get_bool("shell.sandbox_network", False)
        except Exception:
            return False

    @classmethod
    def _synth_etc(cls):
        """Create (once) a minimal passwd/group with NO host users.

        Bound read-only over /etc/passwd + /etc/group so id/git/etc. resolve
        a uid without exposing the host's real account list. Files are
        non-secret (0644); the host dir holding them is never mounted into
        the jail, so only these two paths are visible inside.
        """
        d = cls._synth_etc_dir
        if not (d and os.path.isdir(d)):
            import tempfile
            d = tempfile.mkdtemp(prefix="ag_sandbox_etc_")
            os.chmod(d, 0o755)
            with open(os.path.join(d, "passwd"), "w") as f:
                f.write("root:x:0:0:root:/root:/bin/sh\n"
                        "sandbox:x:1000:1000:sandbox:/workspace:/bin/sh\n"
                        "nobody:x:65534:65534:nobody:/:/usr/sbin/nologin\n")
            with open(os.path.join(d, "group"), "w") as f:
                f.write("root:x:0:\nsandbox:x:1000:\nnogroup:x:65534:\n")
            for fn in ("passwd", "group"):
                os.chmod(os.path.join(d, fn), 0o644)
            cls._synth_etc_dir = d
        return os.path.join(d, "passwd"), os.path.join(d, "group")

    def _sandbox_env(self) -> dict:
        """Clean environment for a sandboxed command (no inherited secrets)."""
        env = {
            "HOME": self.workspace_root,
            "TMPDIR": "/tmp",
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
        for k in self._ENV_PASSTHROUGH:
            v = os.environ.get(k)
            if v and k not in env:
                env[k] = v
        return env

    def _bwrap_argv(self, command_line: str, safe_cwd: str) -> Optional[list]:
        """Build the bubblewrap argv for a sandboxed command (allowlist model).

        Returns None when the command should run unwrapped (not sandboxed,
        sandbox disabled, or bwrap unavailable in 'auto' mode). Raises
        RuntimeError when mode is 'require' but bwrap is missing.

        Mount plan: ALLOWLIST — only the OS runtime dirs + minimal /etc are
        bound read-only; nothing else on the host exists inside the jail. The
        workspace is the single writable real path, bound LAST so no earlier
        mount can shadow it.
        """
        if not self.sandboxed:
            return None
        mode = self._os_sandbox_mode()
        if mode == "off":
            return None
        if not self._BWRAP_PATH:
            if mode == "require":
                raise RuntimeError(
                    "shell.os_sandbox=require but bubblewrap (bwrap) is not "
                    "installed — refusing to run sandboxed shell commands."
                )
            if not ShellExecutionTool._os_sandbox_warned:
                ShellExecutionTool._os_sandbox_warned = True
                logger.warning(
                    "shell.os_sandbox: bwrap not available on this host — "
                    "falling back to regex+rlimit sandbox only (bypassable). "
                    "Install bubblewrap for kernel-enforced isolation."
                )
            return None

        passwd_path, group_path = self._synth_etc()

        argv = [
            self._BWRAP_PATH,
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--unshare-cgroup-try",
        ]
        if not self._sandbox_network_allowed():
            argv += ["--unshare-net"]

        for d in self._RO_ALLOWLIST:
            argv += ["--ro-bind-try", d, d]
        for d in self._ETC_ALLOWLIST:
            argv += ["--ro-bind-try", d, d]

        argv += [
            "--ro-bind", passwd_path, "/etc/passwd",
            "--ro-bind", group_path, "/etc/group",
        ]

        argv += [
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--tmpfs", "/run",
            "--tmpfs", "/var/tmp",
        ]

        for k, v in self._sandbox_env().items():
            argv += ["--setenv", k, v]

        argv += [
            "--bind", self.workspace_root, self.workspace_root,
            "--chdir", safe_cwd,
            "/bin/sh", "-c", command_line,
        ]
        return argv

    _ABS_PATH_PATTERN = re.compile(r'(?:^|\s)(/[a-zA-Z][a-zA-Z0-9_./-]*)')
    
    def _check_path_escape(self, command_line: str) -> str:
        """Check if a command references absolute paths outside the workspace.
        
        Scans for /path/to/something patterns and verifies they're within
        the workspace. Returns error message if escape detected, None if safe.
        """
        for match in self._ABS_PATH_PATTERN.finditer(command_line):
            abs_path = match.group(1).rstrip(';|&')
            
            if abs_path == self.workspace_root:
                continue
            
            real = os.path.realpath(os.path.abspath(abs_path))
            
            if real != self.workspace_root and not real.startswith(self.workspace_root + os.sep):
                return (
                    f"Error: Access denied. Path '{abs_path}' is outside your workspace.\n"
                    f"You can only access files within: {self.workspace_root}\n"
                    f"Use relative paths or paths within your workspace."
                )
        
        if '/..' in command_line or command_line.startswith('..'):
            return (
                "Error: Path traversal ('..') is not allowed in sandboxed mode.\n"
                f"All file operations must stay within: {self.workspace_root}"
            )
        
        return None
        
    def _resolve_cwd(self, cwd: str) -> str:
        """Resolve cwd to absolute path. If sandboxed, must be within workspace."""
        if not os.path.isabs(cwd):
            abs_cwd = os.path.abspath(os.path.join(self.workspace_root, cwd))
        else:
            abs_cwd = os.path.abspath(cwd)
        
        if self.sandboxed:
            real_cwd = os.path.realpath(abs_cwd)
            ws_real = self.workspace_root
            if real_cwd != ws_real and not real_cwd.startswith(ws_real + os.sep):
                raise PermissionError(
                    f"Access denied: Working directory must be within your workspace.\n"
                    f"Requested: {cwd}\n"
                    f"Your workspace: {self.workspace_root}"
                )
        
        if not os.path.isdir(abs_cwd):
            raise FileNotFoundError(f"Directory not found: {abs_cwd}")
        
        return abs_cwd

    def run_command(self, command_line: str, cwd: str = None,
                    wait_ms_before_async: int = 120_000) -> str:
        """Execute a shell command.

        cwd defaults to the SESSION working directory (persisted across
        calls; a `cd` in one command carries into the next). Completes
        within wait_ms_before_async (default 120s, the Claude Code default)
        → returns output; beyond it → promoted to a background task.
        """
        safe_cwd = self._resolve_cwd(cwd or self._session_cwd)

        match = _FORBIDDEN_PATTERN.search(command_line)
        if match:
            return f"Error: Interactive command '{match.group(1)}' not supported."

        tracked_command = self._wrap_for_cwd_tracking(command_line)
        try:
            bwrap_argv = self._bwrap_argv(tracked_command, safe_cwd)
        except RuntimeError as e:
            return f"Error: {e}"

        if self.sandboxed:
            blocked = self._SANDBOX_BLOCKED.search(command_line)
            if blocked:
                return (
                    f"Error: Command '{blocked.group(1).strip()}' is not allowed in sandboxed mode.\n"
                    f"You can only run safe commands within your workspace: {self.workspace_root}"
                )
            if self._SYMLINK_PATTERN.search(command_line):
                return (
                    "Error: Symbolic link creation is not allowed in sandboxed mode.\n"
                    "Symlinks could be used to escape the workspace boundary."
                )
            if bwrap_argv is None:
                path_leak = self._check_path_escape(command_line)
                if path_leak:
                    return path_leak

        timeout_s = max(wait_ms_before_async / 1000.0, 1.0)

        if self.sandboxed:
            env = self._sandbox_env()
        else:
            env = {**os.environ, "PAGER": "cat", "GIT_PAGER": "cat"}

        try:
            result = subprocess.run(
                bwrap_argv if bwrap_argv else tracked_command,
                shell=bwrap_argv is None,
                cwd=safe_cwd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
                preexec_fn=self._sandbox_preexec(),
            )

            output = self._consume_cwd_marker(result.stdout or "")
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            
            if len(output) > self.MAX_OUTPUT_BYTES:
                output = output[:self.MAX_OUTPUT_BYTES] + f"\n...(output truncated at {self.MAX_OUTPUT_BYTES} bytes)"
            
            if not output.strip():
                return f"Command completed (exit code: {result.returncode}) with no output."

            output = output.strip()
            if result.returncode != 0:
                # The model must be able to tell failure from success even
                # when the command printed output.
                output = f"(exit code: {result.returncode})\n{output}"
            return output
            
        except subprocess.TimeoutExpired:
            if self.background_manager:
                # Background tasks run the ORIGINAL command — the cwd marker
                # is only for the synchronous path.
                if bwrap_argv is not None:
                    bg_argv = self._bwrap_argv(command_line, safe_cwd)
                    bg_command = shlex.join(bg_argv)
                else:
                    bg_command = command_line
                return self._spawn_background(bg_command, safe_cwd)
            
            return (f"Command '{command_line[:80]}' exceeded {wait_ms_before_async}ms timeout. "
                    f"No background manager available.")
        except Exception as e:
            return f"Execution error: {e}"

    def _spawn_background(self, command_line: str, cwd: str) -> str:
        """Spawn a command as a background task via the BackgroundTaskManager.
        
        We're in a sync context (called from tool execution thread), so we 
        schedule the async spawn on the event loop.
        """
        try:
            loop = self._loop
            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.background_manager.spawn(command_line, cwd), loop
                )
                return future.result(timeout=5)
            return asyncio.run(
                self.background_manager.spawn(command_line, cwd)
            )
        except Exception as e:
            return f"Failed to spawn background task: {e}"
