import os
import random
import shutil

SOURCE_DIR = "data/raw/all"
IMAGE_DIR = "data/raw/images"
MASK_DIR = "data/raw/masks"

NUM_SAMPLES = 5000

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

files = os.listdir(SOURCE_DIR)

toa_files = [f for f in files if f.endswith("TOA.tif")]
cld_files = [f for f in files if f.endswith("CLD.tif")]

pairs = []

for toa in toa_files:
    
    prefix = toa.split(".TOA")[0]
    
    for cld in cld_files:
        if cld.startswith(prefix):
            pairs.append((toa, cld))
            break

print("Pairs detected:", len(pairs))

random.shuffle(pairs)

pairs = pairs[:NUM_SAMPLES]

for toa, cld in pairs:
    
    shutil.copy(os.path.join(SOURCE_DIR, toa), os.path.join(IMAGE_DIR, toa))
    shutil.copy(os.path.join(SOURCE_DIR, cld), os.path.join(MASK_DIR, cld))

print("Copied", len(pairs), "pairs")