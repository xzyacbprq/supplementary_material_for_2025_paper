import os
import cv2
import glob
from tqdm import tqdm, trange
import numpy as np
import torch
import torchvision.transforms.functional as FF
from torch.utils.data import DataLoader
from torchgeo.datasets import LandCoverAI, stack_samples
from matplotlib import pyplot as plt
from matplotlib import colors
import albumentations as A
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

class LandCoverAI_1():
    """Handles dataset reading, augmentation, and conversion to tensors."""
    def __init__(self, split="train", root = '', ratio=None, transforms=None, seed=42, filter_data = True):
        if split not in {"train", "test", "val"}:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', not '{split}'.")
        
        OUTPUT_DIR = f'{root}output'
        self.num_class   = 5
        self.size        = 512
        self.filter_data = filter_data
        self.mode, self.transforms, self.output_dir = split, transforms, OUTPUT_DIR
        
        with open(os.path.join(root, f"{split}.txt")) as f:
            img_names = f.read().splitlines()

        locs = []
        for x in range(0,512,self.size):
            for y in range(0,512,self.size):
                locs.append([x,x+self.size,y,y+self.size])
        
        self.files = []
        for img_name in img_names:
            for loc in locs:
                self.files.append([img_name,loc])
            
        if ratio:
            print(f"Using {100*ratio:.2f}% of {mode} set --> {int(ratio * len(self.img_names))}/{len(self.img_names)}")
            np.random.seed(seed)
            self.indices = np.random.choice(len(self.img_names), int(ratio * len(self.img_names)), replace=False)
        else:
            print(f"Using the whole {split} set --> {len(self.files)}")
            self.indices = list(range(len(self.files)))
    
    def get_patch(self, img, mask, loc):
        x1, x2, y1, y2 = loc
        return img[x1:x2,y1:y2, :], mask[x1:x2, y1:y2]
        
    def __getitem__(self, item):
        fname, loc = self.files[self.indices[item]]
        img_name = os.path.join(self.output_dir, f"{fname}")
        img  = cv2.imread(f"{img_name}.jpg")
        mask = cv2.imread(f"{img_name}_m.png")[:, :, 1]
        img, mask = self.get_patch(img, mask, loc)
        return {'image' : img, 'mask' : mask}

    def __len__(self):
        return len(self.indices)
        
class LandCoverDataset():
    def __init__(self, split = 'train', img_size = 128, num_class = 5, filter_data = True, aug_data = False, mean = 0.0):
        #root='/media/irfan/TRANSCEND/satellite_data/landcover.ai/'
        root='/data/landcover.ai/'
        self.lc_dataset = LandCoverAI_1(root = root, split = split)
        self.img_size = img_size
        self.num_class = num_class
        self.count = 1
        self.filter_data = filter_data
        self.aug_data    = aug_data
        self.alphas = np.linspace(1.0,1.2,10)
        self.betas  = np.linspace(0,0.2,10)
        self.trans = A.Compose([
            A.OneOf([A.HueSaturationValue(40,40,30,p=1),
                     A.RandomBrightnessContrast(p=1,brightness_limit = 0.2, contrast_limit = 0.5)], p = 0.5),
            A.OneOf([A.RandomRotate90(p=1),
                     A.HorizontalFlip(p=1),
                     A.RandomSizedCrop(min_max_height = (int(0.6*img_size),img_size),size = (img_size,img_size), p =1)], p = 0.5)])
        self.indices = list(range(self.lc_dataset.__len__()))
        self.mean = mean
        if filter_data : self.reset()

    def reset(self):
        for idx in tqdm(range(len(self.indices))):
            ret = self.__getitem__(idx)
            if ret is None:
                break
            
    def process_aux_feats(self, mask):
        _mm  = mask.numpy().copy().astype(np.uint8)
        viz1  = _mm * 0.0
        viz2  = _mm * 0.0
        viz3  = _mm * 0.0
        imgs = []
        kernel = np.ones((3,3),dtype = np.uint8)
        for i in range(self.num_class):
            curr_mask = np.uint8(_mm==i)
            cnts, _  = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            _cnts = []
            for i,cnt in enumerate(cnts):
                area = cv2.contourArea(cnt)
                if area > 0 and area < (10000): #100
                    _cnts.append(cnt)
            edges     = cv2.drawContours(viz1, _cnts, -1, (255), 1)
            proc_mask = cv2.distanceTransform(curr_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            proc_mask = cv2.GaussianBlur(proc_mask, (5,5),1)
            proc_max = proc_mask.max()
            if proc_max > 0.0:
               proc_mask = proc_mask / proc_max
            imgs.append(proc_mask[None])
        
        edges     = cv2.dilate(edges, kernel, iterations =1)
        edges     = cv2.GaussianBlur(edges, (5, 5),1)[None]
        temp      = edges[:,3:-3,3:-3].copy()
        edges     = edges * 0.0
        edges[:,3:-3, 3:-3] = temp
        edge_max  = edges.max()
        if edge_max > 0.0:
           edges     = edges/edge_max
        imgs  = np.concatenate(imgs)#.max(axis=0)
        return imgs, edges
        
    def process_mask_mmseg(self, img, gt, hres, imgs, edges):
        res     = {}
        res['inputs']       = img - self.mean
        res['data_samples'] = {}
        res['data_samples']['gt_sem_seg']  = torch.tensor(gt,dtype=torch.uint8)
        #hres = torch.concat([hres,edges])
        #res['data_samples']['gt_hres_img']  = torch.tensor(hres, dtype=torch.float32)
        res['data_samples']['gt_hres_edge'] = torch.tensor(edges, dtype=torch.float32)
        res['data_samples']['gt_hres_mask'] = torch.tensor(imgs, dtype=torch.float32)
        return res
        
    def __getitem__(self,idx):
        while True:
            if idx > len(self.indices): 
                return None
            if idx >= len(self.indices):
                idx = -1
            smp = self.lc_dataset.__getitem__(self.indices[idx])
            #smp['image'] = smp['image'].permute(1,2,0).numpy()
            #smp['mask']  = smp['mask'].numpy()
            img  = cv2.resize(smp['image'], (self.img_size, self.img_size), interpolation = cv2.INTER_LINEAR)/255.0
            mask = cv2.resize(smp['mask'], (self.img_size, self.img_size), interpolation = cv2.INTER_NEAREST)
            
            if self.filter_data:
                if any([1 for i in range(self.num_class) if np.uint8(mask==i).sum() > (0.80*(self.img_size**2))]):
                    del self.indices[idx]
                    continue
                if not any([1 for i in [1,4] if np.uint8(mask==i).sum() > (0.01*(self.img_size**2))]):
                    del self.indices[idx]
                    continue
            
            if self.aug_data:
                tout = self.trans(image = img.astype(np.float32), mask = mask)
                img, mask = tout['image'], tout['mask']
                
            img  = torch.tensor(img).permute(2,0,1)
            mask = torch.tensor(mask)
    
            imgs, edges = self.process_aux_feats(mask)
            res = self.process_mask_mmseg(img, mask, img, imgs, edges)
            self.count += 1
            break
        return res

    def __len__(self):
        return len(self.indices)

def collate_fn(batch):
    nbatch = []
    for elem in batch:
        #if torch.sum(elem['gt_sem_seg']>0) > 10000 and torch.sum(elem['inputs']==0.0) < 100:
        nbatch.append(elem)
    return stack_samples(nbatch)


class CustomDataLoader():
    def __init__(self, dataset, sampler = None, batch_size = 1, split = 'train'):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.sampler      = sampler
        self.sampler_iter = range(len(self.dataset))
        self.count        = 0
        self.real_count   = 0
        self.len          = len(dataset)
        #self.dataset.len  = self.__len__()
        self.split        = split
        self.full_itr     = False
            
        self.idx = []
        for idx in self.sampler_iter:
            self.idx.append(idx)
            
    def reset(self):
        self.real_count = 0
        self.full_itr = True
        if self.split == 'valid':
            raise StopIteration
           
    def __next__(self):
        nbatch = []
        while 1:
            if self.real_count >= len(self.idx) -1:
                self.reset()
                
            data = self.dataset.__getitem__(self.idx[self.real_count])
            if len(data):
               nbatch.append(data)
               self.real_count += 1
            else:
                del self.idx[self.real_count]
                
            self.count += 1
            if len(nbatch) == self.batch_size:
                nbatch = stack_samples(nbatch)
                return nbatch
                
    def __iter__(self):
        return self
        
    def __len__(self):
        return len(self.idx)//self.batch_size


