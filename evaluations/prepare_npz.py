import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import sys

source = sys.argv[1]
dest = sys.argv[2]

# root = "/mnt/workspace/leijiachen/sample_output/"
# path = os.path.join(root, name)
sample_batch = []
for img in tqdm(glob.glob(source+"/*.png")):
    npz = np.array(Image.open(img).convert("RGB")) # H W C
    sample_batch.append(npz)

sample_batch = np.stack(sample_batch, axis=0)
np.savez(dest, arr_0=sample_batch)