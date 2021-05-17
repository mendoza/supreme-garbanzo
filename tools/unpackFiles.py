from os import listdir
from os.path import join
from tqdm import tqdm
import tarfile
PATH = './data/packs/'
UNPACK_PATH = './data/trainning/'
for pack in tqdm(listdir(PATH)):
    tar = tarfile.open(join(PATH, pack))
    tar.extractall(UNPACK_PATH)
    tar.close()
