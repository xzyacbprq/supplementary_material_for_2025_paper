import os
from copy import deepcopy
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from torchgeo.datasets import NAIP, GeoDataset, RasterDataset, AbovegroundLiveWoodyBiomassDensity, ChesapeakeDE, Landsat, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.datasets.utils import BoundingBox

import numpy as np
from scipy.ndimage import zoom        

from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from rasterio import CRS

img_size = 128
#res = 1.031851524257672e-05

class Landsat8(Landsat):
    """Landsat 8 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS)."""

    filename_glob = 'LC08_*_{}.*'
    default_bands = ('B2', 'B3', 'B4', 'B5', 'B6')
    rgb_bands = ('B4', 'B3', 'B2')
img_dataset = Landsat8('/media/irfan/TRANSCEND/satellite_data/us_cpk_l08_usgs', crs = crs, res = 3*res)

mask = 'cpk'
if mask == 'agb':
    mask_dataset = AbovegroundLiveWoodyBiomassDensity(paths='/media/irfan/TRANSCEND/satellite_data/aglw_cd', 
                                                      crs=None,
                                                      res=None,
                                                      transforms =None,
                                                      download=False,
                                                      cache=True)
if mask == 'cpk':
    #chesapeake_root = '/media/irfan/TRANSCEND/satellite_data/chesapeake'
    chesapeake_train = '/media/irfan/TRANSCEND/satellite_data/chesapeake_dwn/train/2017'
    chesapeake_val   = '/media/irfan/TRANSCEND/satellite_data/chesapeake_dwn/val/2017'
    
    class CPK(RasterDataset):
        filename_glob  = '*/*.tif'
        filename_regex = r'^.{2}_lc_(?P<date>\d{4})_2022-Edition'
        date_format    = '%Y'
        is_image       = False
        cmap = {
         0: (0, 0, 0, 0),
         1: (255, 0, 0, 255),
         2: (0, 255, 0, 255),
         3: (0, 0, 255, 255),
         4: (0, 0, 128, 255),
         5: (0, 128, 0, 255),
         6: (128, 0, 0, 255),
         7: (0, 197, 255, 255),
         8: (38, 115, 0, 255),
         9: (163, 255, 115, 255),
         10: (255, 170, 0, 255),
         11: (156, 156, 156, 255),
         12: (128, 128, 128, 255)}
    
    train_mask_dataset  = CPK(chesapeake_train, crs=crs, res = res)#, res = train_img_dataset.res)
    val_mask_dataset    = CPK(chesapeake_val,   crs=crs, res = res)#,  res = train_img_dataset.res)

aux = 'naip'
if aux == 'naip':
    naip_root   = '/media/irfan/TRANSCEND/satellite_data/us_cpk_naip_5_10_RGBN'
    #aux_dataset = NAIP(naip_root, crs = crs, res = res / 3)
    import pickle
    with open('aux_dataset.pkl','rb') as file:
        aux_dataset = pickle.load(file)

class NDVIDataset(Dataset):
    def __init__(self, split = 'train'):
        if split == 'train':
            self.dataset      = img_dataset  
            self.mask_dataset = train_mask_dataset
            self.aux_dataset  = aux_dataset #& deepcopy(train_mask_dataset)
            
        if split == 'val':
            self.dataset      = img_dataset  
            self.mask_dataset = val_mask_dataset
            self.aux_dataset  = aux_dataset #& deepcopy(val_mask_dataset)
        
        self.mint    = 0.0    #self.dataset.bounds.mint 
        self.maxt    = 1.6e10 #self.dataset.bounds.maxt
        self.index   = (self.dataset.index & self.mask_dataset.index) & self.aux_dataset.index
        self.res     = self.mask_dataset.res
        self.count   = 0
        self.non_zero_thr = 0.1*(img_size**2)
        
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
        
    def normdiff(self, x, y):
        out = np.clip( ((x - y) / ((x + y) + 1e-3)), -0.5, 0.5) + 0.5
        return out
        
    def process_lsat_grns(self, img):
        img = np.nan_to_num(img, nan=0.0, posinf=2**16, neginf=0.0)
        img = cv2.resize(img.transpose(1,2,0), (img_size,img_size), interpolation = cv2.INTER_CUBIC).transpose(2,0,1)
   
        if np.uint8(img.max(axis=0)==0.0).sum() > self.non_zero_thr:
            return [] 
        b, g, r, n, s = img
        ndvi    = self.normdiff(n, r)
        ndwi    = self.normdiff(g, n)
        ndbi    = self.normdiff(s, n)

        img        = torch.tensor(img[:3][[2,1,0]]/2**16,dtype=torch.float32)
        _min, _max = img.min(), img.max()
        img        = (img - _min) / (_max - _min)
        
        out     = np.concatenate([img, ndvi[None,:,:],ndwi[None,:,:]],axis = 0)
        return torch.tensor(out)
        
    def process_naip_rgbn(self, img):
                
        if np.uint8(img.max(axis=0)==0.0).sum() > self.non_zero_thr:
            return []
            
        r, g, b, n = img
        ndvi = self.normdiff(n, r)
        ndwi = self.normdiff(g, n)
        #avg  = ((r + g + b + n)/4.0)/255.0
        img  = torch.tensor(img[:3]/255.0, dtype=torch.float32)
        out  = np.concatenate([img, ndvi[None,:,:], ndwi[None,:,:]])
        
        return torch.tensor(out - 0.5)

    def process_cpk_mask(self, mask):
        mask[mask < 0]      = 0
        mask[mask > 12]     = 0
        new_mask = 0.0 * mask
        new_mask[mask == 1] = 1
        new_mask[mask == 3] = 2 
        new_mask[mask == 5] = 2 # 3
        new_mask[mask == 7] = 3 # 4
        new_mask[mask == 8] = 3 # 4
        #new_mask[mask == 9] = 5
        counts = [np.uint8(new_mask==i).sum()<512 for i in range(4)]
        if any(counts):
            return []
        return new_mask
        
    def process_aux_feats(self, mask):
        _mm  = mask.numpy().copy().astype(np.uint8)
        viz1  = _mm * 0.0
        viz2  = _mm * 0.0
        viz3  = _mm * 0.0
        imgs = []
        
        
        for i in range(4):
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
            proc_mask = proc_mask / proc_mask.max()
            imgs.append(proc_mask[None])

        edges     = cv2.GaussianBlur(edges, (5,5),1)[None]
        edges     = edges/edges.max() 
        imgs  = np.concatenate(imgs)
        return imgs, edges
        
    def __getitem__(self,bbox):

        mask    = self.mask_dataset.__getitem__(bbox)
        mask    = self.process_cpk_mask(mask['mask'][0])
        if not len(mask): return {}
        
        img     = self.dataset.__getitem__(bbox)
        img     = img['image'][:5].numpy().copy() # G, ,R ,N, S
        img     = self.process_lsat_grns(img)
        if not len(img): return {}
            
        hres_img = self.aux_dataset.__getitem__(bbox)
        hres_img = hres_img['image'].numpy().copy()
        hres_img = self.process_naip_rgbn(hres_img)
        if not len(hres_img): return {}
            
        res = {}
        res['mask']     = mask
        res['image']    = img
        imgs, edges = self.process_aux_feats(mask)
        res = self.process_mask_mmseg(img, mask, hres_img, imgs, edges)
        self.count += 1
        return res
        
train_dataset  = NDVIDataset(split = 'train')
val_dataset    = NDVIDataset(split = 'val')
'''
#dataset.__getitem__(0)
minx = 88.0844222351  + 1.0
maxx = 92.6727209818  - 1.0
miny = 20.670883287   + 1.0
maxy = 26.4465255803  - 1.0
bbox = BoundingBox(minx=minx, maxx=maxx, miny=miny, maxy=maxy, mint=comb_dataset.bounds.mint, maxt=comb_dataset.bounds.maxt)
'''
#train_sampler   = RandomGeoSampler(train_dataset, size=256) #, length=4)
train_sampler   = GridGeoSampler(train_dataset, img_size, img_size, units=Units.PIXELS)
val_sampler     = GridGeoSampler(val_dataset, img_size, img_size, units=Units.PIXELS)
#sampler        = RandomGeoSampler(ndvi_dataset, size=64) #, length=4)

def collate_fn(batch):
    nbatch = []
    for elem in batch:
        if torch.sum(elem['gt_sem_seg']>0) > 10000 and torch.sum(elem['inputs']==0.0) < 100:
            nbatch.append(elem)
    return stack_samples(nbatch)
'''
class CustomDataLoader(DataLoader):
    super().__iter__().__next__ = cls.next
    def next(self):
        ret = super().__iter__().__next__()
        if not len(ret['samples']):
            return self.next()
        else:
            return ret
'''

class CustomDataLoader():
    def __init__(self,dataset, sampler = None, batch_size = 1, split = 'train'):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.sampler      = sampler
        self.sampler_iter = iter(sampler)
        self.count        = 0
        self.real_count   = 0
        self.len          = len(sampler)
        #self.dataset.len  = self.__len__()
        self.split        = split
        self.full_itr     = False
        bb     = dataset.dataset.bounds
        if split == 'train' : frac = 0.0
        else: frac = 0.0
            
        x_offset = frac * (bb.maxx - bb.minx)
        y_offset = frac * (bb.maxy - bb.miny)
        
        self.bound = BoundingBox(minx = bb.minx + x_offset, 
                                 maxx = bb.maxx - x_offset, 
                                 miny = bb.miny + y_offset,
                                 maxy = bb.maxy - y_offset,
                                 mint = bb.mint,
                                 maxt = bb.maxt)
        self.bboxes = []
        for bbox in self.sampler_iter:
            if not self.bound.intersects(bbox):
                continue
            self.bboxes.append(bbox)
            
    def reset(self):
        self.real_count = 0
        self.sampler_iter = iter(self.sampler)
        self.full_itr = True
        if self.split == 'valid':
            raise StopIteration
            
    def __next__(self):
        nbatch = []
        while 1:
            if self.real_count >= len(self.bboxes) -1:
                self.reset()
                
            data = self.dataset.__getitem__(self.bboxes[self.real_count])
            if len(data):
               nbatch.append(data)
               self.real_count += 1
            else:
                del self.bboxes[self.real_count]
                
            self.count += 1
            if len(nbatch) == self.batch_size:
                nbatch = stack_samples(nbatch)
                return nbatch
            
    def __iter__(self):
        return self
        
    def __len__(self):
        return len(self.bboxes)
        #if self.len == None: return len(self.sampler)//self.batch_size
        #else: return self.len
            
TrainDataloader = CustomDataLoader(train_dataset, sampler=train_sampler, batch_size = 4)#, collate_fn=collate_fn, num_workers =0)
ValDataloader   = CustomDataLoader(val_dataset, sampler=val_sampler, batch_size = 4, split = 'valid')#, collate_fn=collate_fn, num_workers =0)
        