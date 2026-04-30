# -*- coding: utf-8 -*-
"""Crawl image URLs from search engines using API (no browser, no dependencies)."""

import gzip
import json
import re
import urllib.request
import urllib.parse
from urllib.parse import quote
from concurrent import futures


_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _fetch(url, headers=None, timeout=15):
    """Fetch URL content using urllib (stdlib only), with gzip support."""
    hdrs = dict(_HEADERS)
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8")


# ---------------------------------------------------------------------------
# Baidu
# ---------------------------------------------------------------------------

def _baidu_decode_url(url):
    """Decode Baidu's obfuscated image URLs."""
    in_table = "0123456789abcdefghijklmnopqrstuvw"
    out_table = "7dgjmoru140852vsnkheb963wtqplifca"
    translate_table = str.maketrans(in_table, out_table)
    mapping = {"_z2C$q": ":", "_z&e3B": ".", "AzdH3F": "/"}
    for k, v in mapping.items():
        url = url.replace(k, v)
    return url.translate(translate_table)


def _baidu_get_cookie():
    """Visit baidu.com to obtain a BAIDUID cookie (required by the image API)."""
    try:
        resp = urllib.request.urlopen("https://www.baidu.com", timeout=10)
        for c in (resp.headers.get_all("Set-Cookie") or []):
            if "BAIDUID" in c:
                return c.split(";")[0]
    except Exception:
        pass
    return "BAIDUID=test"


def baidu_get_image_urls(keywords, max_number=10000, face_only=False):
    """Crawl image URLs from Baidu Image Search API."""
    cookie = _baidu_get_cookie()
    extra_headers = {
        "Referer": "https://image.baidu.com/",
        "Cookie": cookie,
    }

    base_url = ("https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj"
                "&ct=201326592&lm=7&fp=result&ie=utf-8&oe=utf-8&st=-1")
    query_url = base_url + "&word={}&queryWord={}".format(
        quote(keywords), quote(keywords))
    query_url += "&face={}".format(1 if face_only else 0)

    # Get total count
    init_data = _fetch(query_url + "&pn=0&rn=30", headers=extra_headers)
    init_json = json.loads(init_data.replace(r"\'", ""), strict=False)
    total_num = init_json.get("listNum", 0)
    if total_num == 0:
        return []

    target_num = min(max_number, total_num)
    crawl_num = min(target_num * 2, total_num)
    batch_size = 30

    def process_batch(batch_no):
        image_urls = []
        url = query_url + "&pn={}&rn={}".format(batch_no * batch_size, batch_size)
        for attempt in range(4):
            try:
                data = _fetch(url, headers=extra_headers)
                break
            except Exception as e:
                if attempt >= 3:
                    print("  Batch {} failed: {}".format(batch_no, e))
                    return image_urls
        try:
            res_json = json.loads(data.replace(r"\'", ""), strict=False)
        except json.JSONDecodeError:
            return image_urls

        for item in res_json.get("data", []):
            if not isinstance(item, dict):
                continue
            # Prefer middleURL (Baidu-hosted, always accessible)
            if "middleURL" in item and item["middleURL"]:
                image_urls.append(item["middleURL"])
            elif "objURL" in item and item["objURL"]:
                from urllib.parse import unquote
                img_url = unquote(_baidu_decode_url(item["objURL"]))
                image_urls.append(img_url)
            elif "replaceUrl" in item and len(item.get("replaceUrl", [])) >= 2:
                image_urls.append(item["replaceUrl"][1].get("ObjURL", ""))
        return [u for u in image_urls if u]

    crawled_urls = []
    batch_count = int((crawl_num + batch_size - 1) / batch_size)
    with futures.ThreadPoolExecutor(max_workers=5) as executor:
        fs = [executor.submit(process_batch, i) for i in range(batch_count)]
        for f in futures.as_completed(fs):
            if f.exception() is None:
                crawled_urls += f.result()
            else:
                print("  Crawl error: {}".format(f.exception()))

    return crawled_urls[:target_num]


# ---------------------------------------------------------------------------
# Bing
# ---------------------------------------------------------------------------

def bing_get_image_urls(keywords, max_number=10000):
    """Crawl image URLs from Bing Image Search API."""
    image_urls = []
    start = 1
    while start <= max_number:
        url = "https://www.bing.com/images/async?q={}&first={}&count=35".format(
            quote(keywords), start)
        try:
            data = _fetch(url)
        except Exception as e:
            print("  Bing fetch error: {}".format(e))
            break
        batch = re.findall(r"murl&quot;:&quot;(.*?)&quot;", data)
        if image_urls and batch and batch[-1] == image_urls[-1]:
            break
        image_urls += batch
        start += len(batch)
        if not batch:
            break
    return image_urls[:max_number]


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def crawl_image_urls(keywords, engine="Baidu", max_number=10000, face_only=False):
    """Crawl image URLs from the specified search engine."""
    print("\nScraping from {} Image Search ...".format(engine))
    print("Keywords: {}".format(keywords))
    print("Target: {}".format(max_number))

    if engine == "Baidu":
        urls = baidu_get_image_urls(keywords, max_number=max_number, face_only=face_only)
    elif engine == "Bing":
        urls = bing_get_image_urls(keywords, max_number=max_number)
    else:
        print("Engine '{}' not supported. Use Baidu or Bing.".format(engine))
        return []

    print("\n=> {} image URLs crawled.\n".format(len(urls)))
    return urls
