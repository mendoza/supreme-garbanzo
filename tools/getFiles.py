from random import sample

import wget

BASE_URL = 'https://ftp.ncbi.nlm.nih.gov/pub/pmc/{}'

with open('./oa_file_list.txt', 'r') as oa:
    lines = sample(oa.readlines(), 5)
    for file in lines:
        url = file.split('\t')[0]
        fileName = url.split('/')[-1]
        wget.download(BASE_URL.format(url), './data/packs/{}'.format(fileName))
