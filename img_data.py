import os

from PIL import Image
from torch.utils import data
from pycocotools.coco import COCO


class Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, path, transform=None):
        "Initialization"
        self.file_names = self.get_filenames(path)
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.file_names)

    def __getitem__(self, index):
        "Generates one sample of data"
        img = Image.open(self.file_names[index]).convert("RGB")
        # Convert image and label to torch tensors
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_filenames(self, data_path):
        images = []
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                if name.rfind("jpg") != -1 or name.rfind("png") != -1:
                    filename = os.path.join(path, name)
                    if os.path.isfile(filename):
                        images.append(filename)
        return images

class CocoDataset(data.Dataset):
    def __init__(self, image_ids, dataDir, dataType, transform=None):
        self.image_ids = image_ids
        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
        self.coco = COCO(self.annFile)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 获取图像信息
        img_id = self.image_ids[idx]
        # print(img_id)
        img = self.coco.loadImgs(img_id)[0]

        # 获取图像文件名
        img_file = os.path.join(self.dataDir, self.dataType, img['file_name'])
        # 读取图像并转换为 tensor
        img_pil = Image.open(img_file).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = img_pil
        # img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        # 获取图像标注信息
        # ann_ids = self.coco.getAnnIds(imgIds=img['id'])
        # anns = self.coco.loadAnns(ann_ids)
        # 将图像数据和标注信息打包为字典并返回
        # sample = {'image': img_tensor, 'annotations': anns}
        return img