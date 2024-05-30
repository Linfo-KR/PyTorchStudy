import urllib
import urllib.request
import zipfile

from tqdm import tqdm

storage_dir = 'D:/PyTorch/Dataset/COCO'

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
        
def download_data(url):
    print(f"Downloading... {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        zip_path, _ = urllib.request.urlretrieve(url, reporthook=t.update_to)
        
    print(f"Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as f:
        for name in tqdm(iterable=f.namelist(), total=len(f.namelist())):
            f.extract(member=name, path=storage_dir)
            

if __name__ == '__main__':
    download_data("http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    download_data("http://images.cocodataset.org/zips/train2017.zip")
    download_data("http://images.cocodataset.org/zips/val2017.zip")