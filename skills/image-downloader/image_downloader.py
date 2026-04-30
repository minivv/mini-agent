#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image Downloader - Search and download images from Baidu/Bing (zero dependencies)."""

import sys
import os
import argparse

# Ensure local modules can be imported regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import crawler
import downloader


def main(argv=None):
    parser = argparse.ArgumentParser(description="Image Downloader (API mode, zero dependencies)")
    parser.add_argument("-k", "--keywords", type=str, required=True,
                        help="Search keywords")
    parser.add_argument("-e", "--engine", type=str, default="Baidu",
                        choices=["Baidu", "Bing"], help="Search engine (default: Baidu)")
    parser.add_argument("-n", "--max-number", type=int, default=10,
                        help="Number of images to download (default: 10)")
    parser.add_argument("-o", "--output", type=str, default="./download_images",
                        help="Output directory (default: ./download_images)")
    parser.add_argument("-fp", "--file-prefix", type=str, default=None,
                        help="Filename prefix (default: engine name)")
    parser.add_argument("-j", "--threads", type=int, default=50,
                        help="Concurrent download threads (default: 50)")
    parser.add_argument("-t", "--timeout", type=int, default=20,
                        help="Download timeout in seconds (default: 20)")
    parser.add_argument("--face-only", action="store_true", default=False,
                        help="Face-only mode (Baidu only)")

    args = parser.parse_args(argv)

    # Crawl 3x URLs to compensate for download failures
    crawl_count = args.max_number * 3
    print("=> Crawling {} image URLs to ensure {} successful downloads ...".format(
        crawl_count, args.max_number))

    urls = crawler.crawl_image_urls(
        args.keywords, engine=args.engine,
        max_number=crawl_count, face_only=args.face_only
    )

    if not urls:
        print("No image URLs found.")
        return

    print("=> Downloading up to {} images ...".format(args.max_number))
    prefix = args.file_prefix or args.engine
    downloader.download_images(
        urls, dst_dir=args.output, file_prefix=prefix,
        max_count=args.max_number, concurrency=args.threads,
        timeout=args.timeout
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
