"""
Phase 3: 沙箱 — 隔离的文件系统 + bash 执行

Sandbox 是 agent 的"安全操作区"。它的核心职责:

  1. 路径隔离 — agent 只能操作 workspace 目录下的文件
  2. 虚拟路径 — agent 看到 /mnt/workspace/foo.txt，
     实际对应 ~/.mini-agent/workspaces/{id}/foo.txt
  3. bash 执行 — 在隔离目录中执行命令

════════════════════════════════════════════════════════════════════════════
路径映射
════════════════════════════════════════════════════════════════════════════
  虚拟路径 (agent 看到的)          实际路径 (磁盘上的)
  /mnt/workspace/foo.txt      →  {base}/{thread_id}/workspace/foo.txt
  /mnt/workspace/sub/bar.py   →  {base}/{thread_id}/workspace/sub/bar.py
  .. (路径穿越尝试)             →  拒绝访问
"""

import glob as glob_mod
import os
import shlex
import subprocess
import fnmatch
from pathlib import Path

# 默认沙箱根目录
DEFAULT_SANDBOX_BASE = Path.home() / ".mini-agent" / "workspaces"


class Sandbox:
    """
    文件系统 + bash 执行沙箱。

    每个对话线程 (thread_id) 拥有独立的沙箱目录，
    agent 只能读写自己沙箱内的文件。
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        base_dir: Path | None = None,
        thread_id: str = "default",
    ):
        """
        创建沙箱。

        两种方式:
          - 从现有目录: Sandbox(workspace_dir="/path/to/workspace")
          - 新建:        Sandbox(thread_id="abc") → ~/.mini-agent/workspaces/abc/workspace/
        """
        if workspace_dir:
            # 从已有的工作目录路径创建（工具内部使用）
            self.workspace = Path(workspace_dir)
            self.workspace.mkdir(parents=True, exist_ok=True)
            self.thread_id = self.workspace.parent.name
            self.base_dir = self.workspace.parent.parent
        else:
            # 新建沙箱目录
            self.base_dir = Path(base_dir or DEFAULT_SANDBOX_BASE)
            self.thread_id = thread_id
            self.workspace = self.base_dir / thread_id / "workspace"
            self.workspace.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 路径解析: 虚拟路径 → 真实路径
    # ------------------------------------------------------------------

    # agent 看到的虚拟挂载点
    VIRTUAL_MOUNTS = {
        "/mnt/workspace": "workspace",
    }

    def _resolve(self, virtual_path: str, must_exist: bool = False) -> Path:
        """
        将 agent 看到的虚拟路径转为磁盘上的真实路径。

        安全检查:
          - 只允许访问挂载点下的路径
          - 拒绝路径穿越 (../../etc/passwd)
        """
        # 找到对应的挂载点
        resolved = None
        for mount, subdir in self.VIRTUAL_MOUNTS.items():
            if virtual_path == mount:
                resolved = self.workspace
                break
            if virtual_path.startswith(mount + "/"):
                relative = virtual_path[len(mount) + 1:]
                resolved = self.workspace / relative
                break

        if resolved is None:
            # 可能是相对路径或不带前缀的路径，视为相对于 workspace
            cleaned = virtual_path.lstrip("/")
            resolved = self.workspace / cleaned

        # 规范化并检查路径穿越
        resolved = resolved.resolve()
        if not str(resolved).startswith(str(self.workspace.resolve())):
            raise PermissionError(f"拒绝访问沙箱外路径: {virtual_path}")

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"文件不存在: {virtual_path}")

        return resolved

    def _display_path(self, real_path: Path) -> str:
        """真实路径 → 虚拟路径 (用于显示给 agent)。"""
        real = str(real_path.resolve())
        workspace = str(self.workspace.resolve())
        if real.startswith(workspace):
            rel = real[len(workspace):].lstrip("/")
            return f"/mnt/workspace/{rel}" if rel else "/mnt/workspace"
        return real

    # ------------------------------------------------------------------
    # 文件操作
    # ------------------------------------------------------------------

    def ls(self, path_str: str = "/mnt/workspace") -> str:
        """列出目录内容，树形格式 (最多 2 层)。"""
        real_path = self._resolve(path_str, must_exist=True)
        return self._tree(real_path, prefix="", max_depth=2)

    def _tree(self, path: Path, prefix: str = "", max_depth: int = 2) -> str:
        """递归树形输出。"""
        if max_depth <= 0:
            return ""
        lines = []
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            return f"{prefix}[拒绝访问]\n"

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            name = entry.name + ("/" if entry.is_dir() else "")
            lines.append(f"{prefix}{connector}{name}")
            if entry.is_dir() and max_depth > 1:
                ext = "    " if is_last else "│   "
                subtree = self._tree(entry, prefix + ext, max_depth - 1)
                if subtree:
                    lines.append(subtree.rstrip("\n"))
        return "\n".join(lines)

    def read_file(self, path_str: str, offset: int = 0, limit: int = 2000) -> str:
        """读取文件内容，支持行号范围。"""
        real_path = self._resolve(path_str, must_exist=True)
        if real_path.is_dir():
            return f"错误: {path_str} 是一个目录"
        try:
            with open(real_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"读取错误: {e}"

        total = len(lines)
        if limit and limit > 0:
            lines = lines[offset: offset + limit]
        elif offset > 0:
            lines = lines[offset:]

        # 加行号
        numbered = []
        for i, line in enumerate(lines, start=offset + 1):
            numbered.append(f"{i:4d}  {line.rstrip()}")
        result = "\n".join(numbered)
        if offset > 0 or (limit and total > offset + limit):
            result += f"\n(共 {total} 行)"
        return result

    def write_file(self, path_str: str, content: str) -> str:
        """创建或覆盖文件，自动创建父目录。"""
        real_path = self._resolve(path_str)
        real_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(real_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"写入错误: {e}"
        return f"已写入: {self._display_path(real_path)} ({len(content)} 字符)"

    def str_replace(self, path_str: str, old_str: str, new_str: str) -> str:
        """替换文件中的字符串 (单次替换)。"""
        real_path = self._resolve(path_str, must_exist=True)
        try:
            with open(real_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"读取错误: {e}"

        if old_str not in content:
            return f"错误: 文件中未找到指定字符串"

        content = content.replace(old_str, new_str, 1)
        try:
            with open(real_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"写入错误: {e}"
        return f"已替换: {self._display_path(real_path)}"

    def glob(self, pattern: str, path_str: str = "/mnt/workspace") -> str:
        """文件模式匹配，如 *.py 或 **/*.py。"""
        real_path = self._resolve(path_str, must_exist=True)
        pattern_full = str(real_path / pattern)
        matches = glob_mod.glob(pattern_full, recursive=True)
        if not matches:
            return f"未找到匹配 {pattern} 的文件"
        lines = []
        for m in sorted(matches):
            p = Path(m)
            if str(p.resolve()).startswith(str(self.workspace.resolve())):
                lines.append(self._display_path(p))
        return "\n".join(lines[:100])  # 最多 100 个

    def grep(self, pattern: str, path_str: str = "/mnt/workspace", glob_pattern: str = "*") -> str:
        """在文件中搜索文本模式，返回匹配行。"""
        base_path = self._resolve(path_str, must_exist=True)
        results = []
        search_root = str(base_path)
        for root, _, files in os.walk(search_root):
            for fname in files:
                if not fnmatch.fnmatch(fname, glob_pattern):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if pattern in line:
                                display = self._display_path(Path(fpath))
                                results.append(f"{display}:{i}: {line.rstrip()[:200]}")
                except Exception:
                    continue
                if len(results) >= 50:
                    break
            if len(results) >= 50:
                break
        return "\n".join(results) if results else f"未找到包含 '{pattern}' 的行"

    # ------------------------------------------------------------------
    # Bash 执行
    # ------------------------------------------------------------------

    def _translate_command(self, command: str) -> str:
        """将命令中的虚拟路径替换为真实路径，这样 bash 能正确访问文件。"""
        translated = command
        for mount, _subdir in self.VIRTUAL_MOUNTS.items():
            # /mnt/workspace/foo.py → {workspace}/foo.py
            # /mnt/workspace → {workspace}
            translated = translated.replace(mount + "/", str(self.workspace) + "/")
            translated = translated.replace(mount, str(self.workspace))
        return translated

    def bash(self, command: str, timeout: int = 60) -> str:
        """
        在沙箱工作目录中执行 bash 命令。

        安全措施:
          - 虚拟路径自动翻译 (/mnt/workspace → 实际目录)
          - 工作在沙箱目录下 (cwd=workspace)
          - 超时机制 (默认 60 秒)
          - 输出截断 (最大 20000 字符)
        """
        # 翻译虚拟路径
        command = self._translate_command(command)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
                env={**os.environ, "HOME": str(self.workspace)},
            )
        except subprocess.TimeoutExpired:
            return f"错误: 命令超时 ({timeout}秒)"
        except Exception as e:
            return f"bash 错误: {e}"

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"[退出码: {result.returncode}]")

        output = "\n".join(output_parts) if output_parts else "(无输出)"
        if len(output) > 20000:
            head = output[:10000]
            tail = output[-10000:]
            output = f"{head}\n... (已截断，共 {len(output)} 字符)\n{tail}"
        return output
