import cv2
import numpy as np
import torch
import torchvision.transforms.functional as FF
from torch.utils.data import DataLoader
from torchgeo.datasets import DeepGlobeLandCover, stack_samples
from matplotlib import pyplot as plt
from matplotlib import colors

from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

class DeepGlobeDataset():
    def __init__(self, split = 'train', img_size = 128, num_class = 7, filter_data = True, aug_data = False):
        root='/media/irfan/TRANSCEND/satellite_data/deep_globe'
        self.lc_dataset = DeepGlobeLandCover(root = root, split = split)
        self.img_size   = img_size
        self.num_class  = num_class
        self.count = 1
        self.filter_data = filter_data
        self.aug_data    = aug_data
        self.alphas = np.linspace(1.0,1.2,10)
        self.betas  = np.linspace(0,0.2,10)

    def process_dpgb_mask(self, mask):
        new_mask = 0.0 * mask
        new_mask[mask == 0] = 1 # 1
        new_mask[mask == 1] = 2 # 2 
        new_mask[mask == 2] = 3 # 3
        new_mask[mask == 3] = 4 # 4
        new_mask[mask == 4] = 5 # 4
        new_mask[mask == 5] = 6
        new_mask[mask == 6] = 0
        #counts = [np.uint8(new_mask==i).sum()<512 for i in range(self.num_class)]
        #if any(counts):
        #    return []
        return new_mask
        
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
        imgs  = np.concatenate(imgs)
        return imgs, edges
        
    def process_mask_mmseg(self, img, gt, hres, imgs, edges):
        res     = {}
        res['inputs']       = img
        res['gt_sem_seg']   = gt 
        res['data_samples'] = SegDataSample()
        res['data_samples'].set_metainfo(dict(img_path=self.count, seg_map_path='', ori_shape=img.shape[1:], img_shape=img.shape[1:]))
        res['data_samples'].gt_sem_seg  = PixelData(data = torch.tensor(gt,dtype=torch.int64))
    
        #hres = torch.concat([hres,edges])
        res['data_samples'].gt_hres_img  = PixelData(data = torch.tensor(hres, dtype=torch.float32))
        res['data_samples'].gt_hres_edge = PixelData(data = torch.tensor(edges, dtype=torch.float32))
        res['data_samples'].gt_hres_mask = PixelData(data = torch.tensor(imgs, dtype=torch.float32))
        return res
        
    def __getitem__(self,idx):
        smp   = self.lc_dataset.__getitem__(idx)
        img   = FF.resize(smp['image'],(self.img_size,self.img_size))/255.0
        mask  = FF.resize(smp['mask'][None],(self.img_size,self.img_size))[0]
        if self.filter_data:
            if any([1 for i in range(self.num_class) if np.uint8(mask==i).sum() > (0.80*(self.img_size**2))]):
                return {}
            if not any([1 for i in [1,4] if np.uint8(mask==i).sum() > (0.01*(self.img_size**2))]):
                return {}
        if self.aug_data:
            aug   = np.random.choice([0,1,2])
            alpha = np.random.choice(self.alphas)
            beta  = np.random.choice(self.betas)
            if aug == 1:    
               img  = torch.flip(img,[1])
               mask = torch.flip(mask,[0])
               #img = torch.clip((img * alpha + beta),0,1)
            if aug == 2:
               img   = torch.flip(img,[2])
               mask  = torch.flip(mask,[1])
               #img = torch.clip((img * alpha + beta),0,1)
        imgs, edges = self.process_aux_feats(mask)
        res = self.process_mask_mmseg(img, mask, img, imgs, edges)
        self.count += 1
        return res

    def __len__(self):
        return self.lc_dataset.__len__()

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
    '''      
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
    '''            
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


