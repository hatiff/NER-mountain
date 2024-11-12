import gdown
import zipfile



gdown.download("https://drive.google.com/uc?export=download&id=1O8sB5-xtjD9g2TFv1K0PZXR4WECqSoda", "pretrained.zip")
with zipfile.ZipFile("pretrained.zip", 'r') as zip_ref:
    zip_ref.extractall('.')