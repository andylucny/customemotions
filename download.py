import requests
import os
from zipfile import ZipFile
from io import BytesIO

def download_espeak():
    if os.path.exists("espeak.exe"):
        return
    print("downloading espeak")
    url = "https://www.agentspace.org/download/espeak.zip"
    response = requests.get(url)
    zip = ZipFile(BytesIO(response.content))
    zip.extractall()
    print("espeak downloaded")

if __name__ == "__main__":
    download_espeak()
    print("done")