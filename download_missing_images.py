import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

# paths
metadata = Path(
    "../datasets/fashion-iq-metadata/image_url"
)

image_dir = Path("fashion-iq/images")
image_dir.mkdir(exist_ok=True)


# lấy danh sách cần có
required = set()

for f in Path("fashion-iq/image_splits").glob("*.json"):
    with open(f) as fp:
        required.update(json.load(fp))


# ảnh đã có
existing = {
    p.stem for p in image_dir.glob("*.jpg")
}


missing = required - existing

print("Missing images:", len(missing))


# đọc URL mapping
url_map = {}

for f in metadata.glob("*.txt"):
    with open(f, errors="ignore") as fp:
        for line in fp:
            parts = line.strip().split()

            if len(parts) >= 2:
                asin = parts[0]
                url = parts[1]
                url_map[asin] = url


print("URLs found:", len(url_map))


success = 0
failed = []


for name in tqdm(missing):

    if name not in url_map:
        failed.append(name)
        continue

    url = url_map[name]

    try:
        r = requests.get(
            url,
            timeout=10
        )

        if r.status_code == 200:
            with open(
                image_dir / f"{name}.jpg",
                "wb"
            ) as f:
                f.write(r.content)

            success += 1
        else:
            failed.append(name)

    except Exception:
        failed.append(name)


print("====================")
print("Downloaded:", success)
print("Failed:", len(failed))


if failed:
    with open("failed_images.txt", "w") as f:
        for x in failed:
            f.write(x+"\n")
