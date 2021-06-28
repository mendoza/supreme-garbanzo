from random import sample
from os import listdir, rmdir, remove
from os.path import join, basename
from tqdm import tqdm
import shutil
import tarfile
import wget
from xml.etree import ElementTree

BASE_URL = 'https://ftp.ncbi.nlm.nih.gov/pub/pmc/{}'

PATH = './try/packs/'
UNPACK_PATH = './try/training/'

with open('./oa_file_list.txt', 'r') as oa:
    lines = oa.readlines()[1:]
    totalGrabbed = 0
    for file in lines:
        url = file.split('\t')[0]
        fileName = url.split('/')[-1]
        wget.download(BASE_URL.format(url), './try/packs/{}'.format(fileName))

        tar = tarfile.open(join(PATH, fileName))
        tar.extractall(UNPACK_PATH)
        tar.close()

        xml = ""
        unpacked_folder_name = fileName.split('.')[0]
        for f in listdir(join(UNPACK_PATH, unpacked_folder_name)):
            if ".nxml" in basename(f):
                xml = join(UNPACK_PATH, unpacked_folder_name, f)

        tree = ElementTree.parse(xml)
        root = tree.getroot()
        abstract = ""
        for p in root.findall("front/article-meta/abstract/sec/p"):
            abstract += " ".join(p.itertext())

        if abstract != '':
            totalGrabbed += 1
        else:
            shutil.rmtree(join(UNPACK_PATH, unpacked_folder_name))
            remove(join(PATH, fileName))

        if totalGrabbed == 1000:
            break
