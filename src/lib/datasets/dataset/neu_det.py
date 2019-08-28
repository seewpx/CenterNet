from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import os
import random
import torch.utils.data as data

class NET_DET(data.Dataset):
  num_classes = 6
  default_resolution = [256, 256]
  mean = np.array([0.503, 0.503, 0.503],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.213, 0.213, 0.213],
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(NET_DET, self).__init__()
    assert split == 'train' or 'val'
    self.data_dir = os.path.join(opt.data_dir, 'NEU-DET')
    self.img_dir = os.path.join(self.data_dir, 'IMAGES')
    #_ann_name = {'train': 'trainval0712', 'val': 'test2007'}
    self.annot_path = os.path.join(
      self.data_dir, # 'ANNOTATIONS', 
      'box_voc_val.json') #.format(_ann_name[split])
    self.max_objs = 50
    self.class_name = ['__background__', "crazing", "inclusion", "patches", 
    "pitted_surface", "rolled-in_scale", "scratches"]
    self._valid_ids = np.arange(1, 7, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt
    self._ran_split_seed = 123
    
    
    print('==> initializing pascal {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    #self.images = sorted(self.coco.getImgIds())
    images_list = sorted(self.coco.getImgIds())
    #random.seed(self._ran_split_seed)
    random.Random(self._ran_split_seed).shuffle(images_list)
    self.all_samples = len(images_list)
    train_size = int(0.9 * self.all_samples)
    val_size = self.all_samples - train_size
    if split == 'train':
        self.num_samples = train_size
        self.images = sorted(images_list[:train_size])
        print('Loaded {} {} samples'.format(split, train_size))
    else:
        self.num_samples = val_size
        self.images =  sorted(images_list[train_size:])
        print('Loaded {} {} samples'.format(split, val_size))
    

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = [[[] for __ in range(self.num_samples)] \
                  for _ in range(self.num_classes + 1)]
    for i in range(self.num_samples):
      img_id = self.images[i]
      for j in range(1, self.num_classes + 1):
        if isinstance(all_bboxes[img_id][j], np.ndarray):
          detections[j][i] = all_bboxes[img_id][j].tolist()
        else:
          detections[j][i] = all_bboxes[img_id][j]
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    os.system('python tools/reval.py ' + \
              '{}/results.json'.format(save_dir))
