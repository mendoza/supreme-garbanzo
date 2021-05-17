# Supreme Garbanzo

Proyecto para sistemas inteligentes 2021 Q2

## Instalación

Este proyecto fue hecho con Python 3.8, y se necesita pipenv:

```ps
pip install pipenv
```

Una vez instalado ambos, se necesita sincronizar los paquetes:

```ps
pipenv sync
```

Como descargar oa_file_list.txt

```ps
curl https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.txt -o oa_file_list.txt
```

Como obtener el trainning data; Esto obtendra 1000 lineas random del archivo oa_file_list.txt y descargara los archivos 
```ps
mkdir data/packs
python ./tools/getFiles.py
mkdir data/trainning
python ./tools/unpackFiles.py
```

## Correr el proyecto

Para poder trabajar en el proyecto de la manera más optima, seleccionar el interpretador de pipenv en la parte inferior izquierda de VS Code.

```ps
# aun no se...
```
