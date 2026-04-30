---
name: image-downloader
description: 从百度/必应图片搜索下载指定数量的图片到指定路径。当用户需要批量下载图片、搜索并保存网络图片、或需要获取特定主题的图片素材时使用此 skill。
---

# Image Downloader

## 概述

此 skill 可根据关键词从百度/必应图片搜索并批量下载图片。**纯 Python 标准库实现，零依赖**，无需 pip install 或 venv。

## 使用方法

**路径说明**：`image_downloader.py` 脚本位于本 SKILL.md 文件的**同级目录**下。请根据你读取到本文件的实际路径，将下面的 `<SKILL_DIR>` 替换为该目录的绝对路径。

```bash
python3 "<SKILL_DIR>/image_downloader.py" -k "关键词" -o "输出目录" -n 数量
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-k` | (必填) | 搜索关键词 |
| `-e` | Baidu | 搜索引擎 (Baidu / Bing) |
| `-o` | ./download_images | 输出目录 |
| `-n` | 10 | 下载图片数量 |
| `-fp` | (引擎名) | 文件名前缀 |
| `-j` | 50 | 并发下载线程数 |
| `-t` | 20 | 超时秒数 |
| `--face-only` | false | 仅人脸模式 (仅百度) |

### 示例

**下载 10 张猫咪图片到 ./cats 目录：**
```bash
python3 "<SKILL_DIR>/image_downloader.py" -k "可爱猫咪" -o ./cats
```

**从 Bing 下载 20 张风景图片：**
```bash
python3 "<SKILL_DIR>/image_downloader.py" -k "山水风景" -e Bing -o ./landscape -n 20
```

**下载 50 张狗狗图片，自定义前缀：**
```bash
python3 "<SKILL_DIR>/image_downloader.py" -k "金毛犬" -o ./dogs -n 50 -fp dog
```
