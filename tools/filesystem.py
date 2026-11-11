import os
import re
import stat


class FilesystemTools:
    """Filesystem operations with optional sandbox enforcement.
    
    Security model:
    - sandboxed=False (CLI/local): Unrestricted access (OS user's permissions).
    - sandboxed=True  (Cloud/Web): JAILED to workspace_root. All paths are
      resolved through os.path.realpath() to defeat symlink traversal, then
      checked against the workspace boundary. Any path that escapes the
      workspace is rejected.
    
    Defense-in-depth layers:
      1. os.path.abspath() — normalizes ../ traversal in path strings
      2. os.path.realpath() — resolves symlinks to true filesystem location
      3. Boundary check — resolved path must start with workspace_root + /
      4. Binary detection — prevents reading binary files
      5. Output capping — prevents token exhaustion
    """
    
    MAX_OUTPUT_LINES = 800
    MAX_GREP_MATCHES = 50
    
    def __init__(self, workspace_root: str, sandboxed: bool = False, max_size_bytes: int = 0):
        self.workspace_root = os.path.realpath(os.path.abspath(workspace_root))
        self.sandboxed = sandboxed
        self.max_size_bytes = max_size_bytes
        
    def _resolve_path(self, path: str) -> str:
        """Resolve a path to absolute. Relative paths joined with workspace root.
        
        When sandboxed:
          - Resolves symlinks via realpath (blocks symlink escape)
          - Checks resolved path is within workspace_root
          - Returns error sentinel if path escapes the sandbox
        """
        if not os.path.isabs(path):
            resolved = os.path.abspath(os.path.join(self.workspace_root, path))
        else:
            resolved = os.path.abspath(path)
        
        if self.sandboxed:
            real = os.path.realpath(resolved)
            ws_real = self.workspace_root
            
            if real != ws_real and not real.startswith(ws_real + os.sep):
                return None
        
        return resolved
    
    def _access_denied_msg(self, path: str) -> str:
        """Standard access denied message for sandbox violations."""
        return (
            f"Error: Access denied. Path '{path}' is outside your workspace.\n"
            f"You can only access files within: {self.workspace_root}\n"
            f"Use relative paths or paths starting with {self.workspace_root}"
        )

    def _is_binary(self, path: str) -> bool:
        """Detect binary files by reading first 8KB and checking for null bytes."""
        try:
            with open(path, 'rb') as f:
                chunk = f.read(8192)
            return b'\x00' in chunk
        except IOError:
            return True


    def view_file(self, absolute_path: str, start_line: int = None, end_line: int = None) -> str:
        """Read a file's contents with line numbers. Output capped at 800 lines."""
        path = self._resolve_path(absolute_path)
        if path is None:
            return self._access_denied_msg(absolute_path)
        
        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        if os.path.isdir(path):
            return f"Error: {path} is a directory. Use list_dir instead."
        if self._is_binary(path):
            return f"Error: {path} is a binary file."
            
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()
        except IOError as e:
            return f"Error reading {path}: {e}"
        
        total_lines = len(all_lines)
        
        if start_line or end_line:
            start_idx = max(0, (start_line or 1) - 1)
            end_idx = min(total_lines, end_line) if end_line else total_lines
            lines = all_lines[start_idx:end_idx]
            first_line_num = start_idx + 1
        else:
            lines = all_lines
            first_line_num = 1
        
        truncated = False
        if len(lines) > self.MAX_OUTPUT_LINES:
            lines = lines[:self.MAX_OUTPUT_LINES]
            truncated = True
        
        output = []
        for i, line in enumerate(lines):
            output.append(f"{first_line_num + i}: {line.rstrip()}")
        
        header = f"File: {path} | Total: {total_lines} lines | Showing: {first_line_num}-{first_line_num + len(lines) - 1}"
        if truncated:
            header += f" (TRUNCATED at {self.MAX_OUTPUT_LINES} lines)"
        
        return header + "\n" + "\n".join(output)


    def write_to_file(self, target_file: str, code_content: str, overwrite: bool = False) -> str:
        """Write content to a file, creating parent directories as needed."""
        path = self._resolve_path(target_file)
        if path is None:
            return self._access_denied_msg(target_file)
        
        if os.path.exists(path) and not overwrite:
            return f"Error: {path} already exists. Set overwrite=true to replace."
        
        if self.sandboxed and self.max_size_bytes > 0:
            content_size = len(code_content.encode('utf-8'))
            current_size = self._get_workspace_size()
            if current_size + content_size > self.max_size_bytes:
                quota_mb = self.max_size_bytes / (1024 * 1024)
                used_mb = current_size / (1024 * 1024)
                return (
                    f"Error: Workspace quota exceeded.\n"
                    f"Quota: {quota_mb:.1f} MB | Used: {used_mb:.1f} MB | "
                    f"Write size: {content_size / 1024:.1f} KB\n"
                    f"Free up space or contact admin to increase your quota."
                )
        
        parent_dir = os.path.dirname(path)
        if self.sandboxed:
            real_parent = os.path.realpath(parent_dir)
            if (real_parent != self.workspace_root and
                    not real_parent.startswith(self.workspace_root + os.sep)):
                return self._access_denied_msg(target_file)
        
        os.makedirs(parent_dir, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        line_count = code_content.count('\n') + (1 if code_content and not code_content.endswith('\n') else 0)
        return f"Successfully wrote {line_count} lines to {path}"
    
    def _get_workspace_size(self) -> int:
        """Calculate total size of workspace in bytes (for quota check)."""
        total = 0
        for dirpath, _, filenames in os.walk(self.workspace_root):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total


    def list_dir(self, directory_path: str) -> str:
        """List directory contents with type, size, and child count.
        
        Output format matches Agent's list_dir:
        - Files: name, size in bytes
        - Directories: name, number of direct children
        """
        path = self._resolve_path(directory_path)
        if path is None:
            return self._access_denied_msg(directory_path)
        
        if not os.path.exists(path):
            return f"Error: Directory not found: {path}"
        if not os.path.isdir(path):
            return f"Error: {path} is a file, not a directory."
        
        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            return f"Error: Permission denied reading {path}"
        
        dirs = []
        files = []
        
        for item in items:
            if item.startswith('.') and item in ('.git', '.DS_Store', '__pycache__', '.ag_data'):
                continue
            full = os.path.join(path, item)
            try:
                st = os.stat(full)
                if stat.S_ISDIR(st.st_mode):
                    try:
                        child_count = len(os.listdir(full))
                    except PermissionError:
                        child_count = -1
                    dirs.append(f"  {item}/ ({child_count} children)")
                else:
                    size = st.st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1048576:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/1048576:.1f}MB"
                    files.append(f"  {item} ({size_str})")
            except OSError:
                files.append(f"  {item} (unreadable)")
        
        output = [f"Directory: {path}"]
        if dirs:
            output.append(f"Directories ({len(dirs)}):")
            output.extend(dirs)
        if files:
            output.append(f"Files ({len(files)}):")
            output.extend(files)
        if not dirs and not files:
            output.append("(empty)")
        
        return "\n".join(output)


    MAX_GLOB_RESULTS = 100
    _SKIP_DIRS = frozenset({'.git', '__pycache__', 'node_modules', '.ag_data',
                            '.venv', 'venv', '.pytest_cache'})

    def glob_files(self, pattern: str, path: str = None) -> str:
        """Find files whose path matches a glob pattern, newest-modified first.

        The fast way to LOCATE files by name/extension before reading them —
        use instead of recursive list_dir walks.
        """
        import fnmatch
        base = self._resolve_path(path) if path else self.workspace_root
        if base is None:
            return self._access_denied_msg(path)
        if not os.path.isdir(base):
            return f"Error: {base} is not a directory."

        matches = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in self._SKIP_DIRS]
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, base)
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fname, pattern):
                    try:
                        mtime = os.path.getmtime(full)
                    except OSError:
                        mtime = 0.0
                    matches.append((mtime, full))

        if not matches:
            return f"No files match '{pattern}' under {base}."

        matches.sort(reverse=True)
        shown = [full for _, full in matches[:self.MAX_GLOB_RESULTS]]
        header = f"{len(matches)} files match '{pattern}' (newest first)"
        if len(matches) > self.MAX_GLOB_RESULTS:
            header += f", showing {self.MAX_GLOB_RESULTS} — narrow the pattern for the rest"
        return header + ":\n" + "\n".join(shown)

    def grep_search(self, search_path: str, query: str, is_regex: bool = False,
                    case_insensitive: bool = False, includes: str = None,
                    context_lines: int = 0, files_with_matches: bool = False,
                    max_results: int = 0) -> str:
        """Search files for pattern matches.

        Modes: matching lines (default, optional ±context_lines of context),
        or files_with_matches (just the file paths — cheap way to scope a
        symbol before reading). max_results raises the default cap (≤200);
        a truncation marker is emitted when more matches exist.
        """
        path = self._resolve_path(search_path)
        if path is None:
            return self._access_denied_msg(search_path)

        if not os.path.exists(path):
            return f"Error: Path not found: {path}"

        flags = re.IGNORECASE if case_insensitive else 0
        if is_regex:
            try:
                pattern = re.compile(query, flags)
            except re.error as e:
                return f"Error: Invalid regex: {e}"
        else:
            pattern = re.compile(re.escape(query), flags)

        include_globs = None
        if includes:
            include_globs = [g.strip() for g in includes.split(',')]

        cap = min(max_results, 200) if max_results and max_results > 0 \
            else self.MAX_GREP_MATCHES
        ctx = max(0, min(int(context_lines or 0), 10))

        results = []
        matched_files = []
        more_exist = [False]

        def _search_file(filepath: str):
            if self._is_binary(filepath):
                return
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
            except (IOError, UnicodeDecodeError):
                return
            for idx, line in enumerate(lines):
                if not pattern.search(line):
                    continue
                if files_with_matches:
                    matched_files.append(filepath)
                    return
                if len(results) >= cap:
                    more_exist[0] = True
                    return
                if ctx:
                    lo = max(0, idx - ctx)
                    hi = min(len(lines), idx + ctx + 1)
                    block = []
                    for j in range(lo, hi):
                        sep = ':' if j == idx else '-'
                        block.append(f"{filepath}{sep}{j + 1}{sep} "
                                     f"{lines[j].rstrip()[:200]}")
                    results.append("\n".join(block))
                else:
                    results.append(f"{filepath}:{idx + 1}: {line.strip()[:200]}")

        if os.path.isfile(path):
            _search_file(path)
        else:
            import fnmatch
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in self._SKIP_DIRS]
                for fname in sorted(files):
                    if files_with_matches:
                        if len(matched_files) >= cap:
                            more_exist[0] = True
                            break
                    elif len(results) >= cap:
                        break
                    if include_globs and not any(
                            fnmatch.fnmatch(fname, g) for g in include_globs):
                        continue
                    _search_file(os.path.join(root, fname))

        if files_with_matches:
            if not matched_files:
                return "No files contain a match."
            header = f"{len(matched_files)} files contain matches"
            if more_exist[0]:
                header += " (more exist — narrow the search)"
            return header + ":\n" + "\n".join(matched_files)

        if not results:
            return "No matches found."

        header = f"Found {len(results)} matches"
        if more_exist[0] or len(results) >= cap:
            header += (f" (capped at {cap}; more exist — use a more specific "
                       f"query, includes filter, or files_with_matches)")
        sep = "\n--\n" if ctx else "\n"
        return header + ":\n" + sep.join(results)


    def replace_file_content(self, target_file: str, target_content: str,
                             replacement_content: str, start_line: int = None,
                             end_line: int = None,
                             replace_all: bool = False) -> str:
        """Replace text in a file.

        Default: target_content must match exactly ONCE (use start_line/
        end_line to disambiguate). replace_all=True replaces every
        occurrence. The result echoes the edited region so a follow-up
        re-read is unnecessary.
        """
        path = self._resolve_path(target_file)
        if path is None:
            return self._access_denied_msg(target_file)

        if not os.path.exists(path):
            return f"Error: File not found: {path}"

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        change_pos = None
        replaced = 1
        if start_line and end_line:
            lines = content.split('\n')
            search_region = '\n'.join(lines[start_line-1:end_line])

            if target_content not in search_region:
                return f"Error: target_content not found in lines {start_line}-{end_line}."

            count = search_region.count(target_content)
            if count > 1 and not replace_all:
                return f"Error: target_content found {count} times in range. Must be unique (or set replace_all)."

            replaced = count if replace_all else 1
            new_region = search_region.replace(
                target_content, replacement_content, -1 if replace_all else 1)
            change_pos = len('\n'.join(lines[:start_line-1])) + search_region.find(target_content)
            lines[start_line-1:end_line] = new_region.split('\n')
            new_content = '\n'.join(lines)
        else:
            count = content.count(target_content)
            if count == 0:
                return "Error: target_content not found in file."
            if count > 1 and not replace_all:
                return (f"Error: target_content found {count} times. Use "
                        f"start_line/end_line to disambiguate, or set replace_all=true.")
            replaced = count if replace_all else 1
            change_pos = content.find(target_content)
            new_content = content.replace(
                target_content, replacement_content, -1 if replace_all else 1)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        summary = f"Replaced {replaced} occurrence(s) in {path}."
        if replaced == 1 and change_pos is not None:
            line_no = new_content[:change_pos].count('\n') + 1
            new_lines = new_content.split('\n')
            lo = max(0, line_no - 3)
            hi = min(len(new_lines),
                     line_no + replacement_content.count('\n') + 2)
            snippet = "\n".join(f"{i + 1}: {new_lines[i]}" for i in range(lo, hi))
            summary += f" Edited region now reads:\n{snippet}"
        return summary


    def multi_replace_file_content(self, target_file: str, replacements: str) -> str:
        """Apply multiple non-contiguous replacements to a file.
        
        replacements: JSON string of array [{target, replacement}, ...]
        Processes in reverse position order to maintain correct offsets.
        """
        import json as _json
        path = self._resolve_path(target_file)
        if path is None:
            return self._access_denied_msg(target_file)
        
        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        
        try:
            chunks = _json.loads(replacements) if isinstance(replacements, str) else replacements
        except _json.JSONDecodeError as e:
            return f"Error: Invalid JSON in replacements: {e}"
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        positions = []
        for i, chunk in enumerate(chunks):
            target = chunk.get("target", "")
            pos = content.find(target)
            if pos == -1:
                return f"Error: Chunk {i} target_content not found in file."
            positions.append((pos, chunk))
        
        positions.sort(key=lambda x: x[0], reverse=True)
        
        for pos, chunk in positions:
            target = chunk["target"]
            replacement = chunk.get("replacement", "")
            content = content[:pos] + replacement + content[pos + len(target):]
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully applied {len(chunks)} replacements to {path}"
