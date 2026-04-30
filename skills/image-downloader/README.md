# Image Downloader

从百度图片搜索下载图片的命令行工具。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python image_downloader.py --keywords "搜索关键词" --output 输出目录
```

## 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--keywords` | `-k` | (必填) | 搜索关键词 |
| `--output` | `-o` | ./download_images | 输出目录 |
| `--max-number` | `-n` | 10 | 下载图片数量 |
| `--file-prefix` | `-fp` | Baidu | 文件名前缀 |
| `--engine` | `-e` | Baidu | 搜索引擎 (Google/Bing/Baidu) |
| `--driver` | `-d` | api | 驱动模式 (api/chrome/chrome_headless) |

## 示例

```bash
# 下载 10 张 CSGO 赛事图片
python image_downloader.py -k "CSGO赛事" -o ./output

# 下载 20 张图片，自定义文件名前缀
python image_downloader.py -k "CSGO赛事" -n 20 -fp event -o ./output
```
