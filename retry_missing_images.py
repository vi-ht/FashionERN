import time
import requests
from pathlib import Path
from tqdm import tqdm

metadata = Path(
    "../datasets/fashion-iq-metadata/image_url"
)

image_dir = Path("fashion-iq/images")
missing_file = Path("missing_images.txt")

# Read missing image IDs
missing = [
    x.strip()
    for x in missing_file.read_text().splitlines()
    if x.strip()
]

# Build URL map
url_map = {}

for f in metadata.glob("*.txt"):
    with open(f, errors="ignore") as fp:
        for line in fp:
            parts = line.strip().split()

            if len(parts) >= 2:
                url_map[parts[0]] = parts[1]

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/139 Safari/537.36"
    )
})

success = 0
failed = []

for name in tqdm(missing):

    save_path = image_dir / f"{name}.jpg"

    # Skip if already downloaded
    if save_path.exists():
        continue

    if name not in url_map:
        failed.append((name, "NO_URL"))
        continue

    url = url_map[name]

    downloaded = False

    # Retry 3 times
    for attempt in range(3):

        try:
            r = session.get(
                url,
                timeout=30
            )

            if (
                r.status_code == 200
                and r.headers.get("Content-Type", "").startswith("image")
            ):
                with open(save_path, "wb") as f:
                    f.write(r.content)

                success += 1
                downloaded = True
                break

            else:
                error = f"HTTP_{r.status_code}"

        except Exception as e:
            error = type(e).__name__

        time.sleep(1)

    if not downloaded:
        failed.append((name, error))


print()
print("=" * 50)
print("Retry completed")
print("=" * 50)
print("Downloaded:", success)
print("Still failed:", len(failed))

if failed:
    with open("failed_images_final.txt", "w") as f:
        for name, error in failed:
            f.write(f"{name}\t{error}\n")
