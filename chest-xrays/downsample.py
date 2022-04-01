import os
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

os.mkdir('processed')

trans = T.Resize((256, 256), interpolation=Image.LANCZOS)

for fname in tqdm(os.listdir('all-images')):
    path = os.path.join('all-images', fname)
    img = Image.open(path)
    img = trans(img)
    proc_path = os.path.join('processed', fname)
    img.save(proc_path)
