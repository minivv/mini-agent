"""
Phase 6: Skills — 可复用的知识模块

════════════════════════════════════════════════════════════════════════════
什么是 Skills?
════════════════════════════════════════════════════════════════════════════

Skills 是预定义的知识模块，告诉 Agent 如何处理特定类型的任务。

打个比方:
  - Agent 是一个"通才员工"，什么都能干一点
  - Skills 是"操作手册"——当遇到特定任务时，先读对应的手册
  - 比如遇到"代码审查"任务，先读 code-review 的操作手册，
    然后按照手册的步骤来做，而不是自己随便发挥

Skill 的文件格式:
  skills/
    code-review/
      SKILL.md        ← 技能描述文件（YAML frontmatter + Markdown 正文）
      references/     ← 参考资料（可选）
      templates/      ← 模板文件（可选）

SKILL.md 的 YAML frontmatter:
  ---
  name: code-review
  description: 用这个技能来进行代码审查...
  ---
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Skill:
    """一个技能的数据结构。"""
    name: str
    description: str
    skill_file: Path       # SKILL.md 的路径
    category: str = "public"


# ---------------------------------------------------------------------------
# Skill 文件解析
# ---------------------------------------------------------------------------

def parse_skill_file(skill_file: Path) -> Skill | None:
    """
    解析 SKILL.md 文件，提取 YAML frontmatter 和技能信息。

    文件格式:
      ---
      name: skill-name
      description: 技能描述
      ---
      # 正文内容...
    """
    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception:
        return None

    # 提取 YAML frontmatter（--- 之间的内容）
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None

    if not isinstance(frontmatter, dict):
        return None

    name = frontmatter.get("name", "")
    description = frontmatter.get("description", "")

    if not name or not description:
        return None

    return Skill(
        name=str(name),
        description=str(description),
        skill_file=skill_file,
        category="public",
    )


# ---------------------------------------------------------------------------
# Skill 发现
# ---------------------------------------------------------------------------

def load_skills(skills_dir: str | Path | None = None) -> list[Skill]:
    """
    扫描 skills/ 目录，发现所有技能。

    遍历 skills_dir 下所有子目录，找到包含 SKILL.md 的目录，
    解析为 Skill 对象。

    参数:
      skills_dir: skills 目录路径，默认为项目根目录下的 skills/
    """
    if skills_dir is None:
        skills_dir = Path(__file__).parent.parent.parent / "skills"
    else:
        skills_dir = Path(skills_dir)

    if not skills_dir.is_dir():
        return []

    skills: dict[str, Skill] = {}

    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.exists():
            continue
        skill = parse_skill_file(skill_file)
        if skill:
            skills[skill.name] = skill

    return sorted(skills.values(), key=lambda s: s.name)


# ---------------------------------------------------------------------------
# 系统提示词生成
# ---------------------------------------------------------------------------

def get_skills_prompt_section(skills: list[Skill], skills_dir: str | Path) -> str:
    """
    生成注入到系统提示词中的 Skills 部分。

    采用"渐进式加载"模式:
      1. 在系统提示词中列出所有可用技能的名称、描述和路径
      2. Agent 遇到匹配的任务时，用 read_file 读取对应的 SKILL.md
      3. 然后按照技能中的工作流程来执行任务

    这样做的优势:
      - 系统提示词不会太长（只放摘要，不放完整内容）
      - Agent 只加载需要的技能（按需加载）
      - 添加新技能不需要改代码，只要放到 skills/ 目录即可
    """
    if not skills:
        return ""

    skills_dir = Path(skills_dir)

    lines = [
        "<skill_system>",
        "你可以使用技能（Skills）来处理特定类型的任务。",
        "每个技能是一个预定义的工作流程，包含优化的步骤和最佳实践。",
        "",
        "**渐进式加载模式:**",
        "1. 当用户的请求匹配某个技能时，用 read_file 读取该技能的 SKILL.md",
        "2. 理解技能的工作流程和指令",
        "3. 按照技能的步骤来执行任务",
        "",
        "**技能目录:** /mnt/skills",
        "",
        "<available_skills>",
    ]

    for skill in skills:
        # 计算技能在虚拟挂载点下的路径
        relative = skill.skill_file.relative_to(skills_dir)
        location = f"/mnt/skills/{relative}"
        lines.append(f"    <skill>")
        lines.append(f"        <name>{skill.name}</name>")
        lines.append(f"        <description>{skill.description}</description>")
        lines.append(f"        <location>{location}</location>")
        lines.append(f"    </skill>")

    lines.append("</available_skills>")
    lines.append("</skill_system>")
    lines.append("")

    return "\n".join(lines)
