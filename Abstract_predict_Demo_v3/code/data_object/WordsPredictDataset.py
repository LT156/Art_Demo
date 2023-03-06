import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class WordsPredictDataset(Dataset):
    def __init__(self, image_files,  abs_features=None,img_transform=None, rgb_only=True):
        super(WordsPredictDataset, self).__init__()
        self.image_files = image_files
        self.abs_features = abs_features
        self.img_transform = img_transform
        self.rgb_only = rgb_only

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])
        abs_features = np.array(self.abs_features[index]).astype(dtype=np.float32)

        if self.rgb_only and img.mode is not 'RGB':
            img = img.convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)
        res = {'image': img,'abstract_data':abs_features, 'index': index}
        return res

    def __len__(self):
        return len(self.image_files)