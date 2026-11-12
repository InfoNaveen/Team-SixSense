"""
Download raw 5-minute Delhi electricity demand and weather dataset from Kaggle.
"""

import os
import shutil
import kagglehub

def download():
    path = kagglehub.dataset_download("yug201/delhi-5-minute-electricity-demand-for-forecasting")
    print("Cached at:", path)
    os.makedirs("data/raw", exist_ok=True)
    for f in os.listdir(path):
        src = os.path.join(path, f)
        dst = os.path.join("data/raw", f)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, "data/raw")
    print("Files in data/raw:", os.listdir("data/raw"))

if __name__ == "__main__":
    download()

