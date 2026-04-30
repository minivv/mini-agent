# -*- coding: utf-8 -*-
"""Download images from URLs using only Python stdlib."""

import os
import socket
import concurrent.futures
import urllib.request
import urllib.parse


def _detect_image_format(data):
    """Detect image format from magic bytes. Returns extension string or None."""
    if len(data) < 12:
        return None
    if data[:3] == b"\xFF\xD8\xFF":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1A\n":
        return "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] == b"BM":
        return "bmp"
    return None


def _make_headers(url):
    """Build request headers with appropriate Referer."""
    parsed = urllib.parse.urlparse(url)
    referer = "{}://{}".format(parsed.scheme, parsed.netloc)
    return {
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        "Referer": referer,
    }


def download_image(image_url, dst_dir, file_name, timeout=20):
    """Download a single image, validate via magic bytes, save with correct extension.

    Returns (True, extension) on success, False on failure.
    """
    for attempt in range(3):
        try:
            headers = _make_headers(image_url)
            req = urllib.request.Request(image_url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()

            fmt = _detect_image_format(data)
            if fmt is None:
                print("## Err: Not a valid image  {}".format(image_url))
                return False

            final_name = "{}.{}".format(file_name, fmt)
            final_path = os.path.join(dst_dir, final_name)
            with open(final_path, "wb") as f:
                f.write(data)

            print("## OK:    {}  {}".format(final_name, image_url))
            return (True, fmt)

        except Exception as e:
            if attempt >= 2:
                print("## Fail:  {}  {}".format(image_url, e))
                return False
    return False


def download_images(image_urls, dst_dir, file_prefix="img", max_count=None,
                    concurrency=50, timeout=20):
    """Download images concurrently and rename sequentially.

    Returns the number of successfully downloaded images.
    """
    socket.setdefaulttimeout(timeout)
    os.makedirs(dst_dir, exist_ok=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {}
        for idx, url in enumerate(image_urls):
            temp_name = "_temp_{}_{}".format(file_prefix, idx)
            future = executor.submit(download_image, url, dst_dir, temp_name, timeout)
            future_map[future] = idx

        for future in concurrent.futures.as_completed(future_map, timeout=300):
            idx = future_map[future]
            try:
                result = future.result()
                if result:
                    results.append((idx, result[1]))
            except Exception as e:
                print("## Exception: {}".format(e))

    # Sort by original crawl order, rename sequentially
    results.sort()
    kept = 0

    for idx, ext in results:
        old_path = os.path.join(dst_dir, "_temp_{}_{}.{}".format(file_prefix, idx, ext))
        if not os.path.exists(old_path):
            continue
        kept += 1
        if max_count is not None and kept > max_count:
            os.remove(old_path)
        else:
            new_path = os.path.join(dst_dir, "{}{}.{}".format(file_prefix, kept, ext))
            os.rename(old_path, new_path)

    total = len(results)
    if max_count and total > max_count:
        print("\n=> Downloaded {}, kept {} (as requested)".format(total, max_count))
    else:
        print("\n=> Successfully downloaded {} out of {} URLs".format(kept, len(image_urls)))

    return kept
