# -*- coding: utf-8 -*-
'''
@File    :   visualizer.py
@Time    :   2022/04/05 11:39:33
@Author  :   Shilong Liu 
@Contact :   liusl20@mail.tsinghua.edu.cn; slongliu86@gmail.com
Modified from COCO evaluator
'''

import os, sys
from textwrap import wrap
import torch
import numpy as np
import cv2
import datetime

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
from matplotlib import transforms
from matplotlib.font_manager import FontProperties

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)

class ColorMap():
    def __init__(self, basergb=[255,255,0]):
        self.basergb = np.array(basergb)
    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1) # h, w, 3
        attn1 = attnmap.copy()[..., None] # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res

import matplotlib.patches as mpatches

class COCOVisualizer():
    def __init__(self) -> None:
        pass

    def visualize(self, img, tgt, dpi=100, savedir=None, it = 0, show_in_console=True,fontsize = 14,offset =0.1,Chinese=False,Astro = False ):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = '5'
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0)
        ax.imshow(img)
        if tgt is not None:
            self.addtgt(tgt,fontsize = fontsize, offset = offset,Chinese=Chinese, Astro = Astro)
        
        plt.axis('off')

        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)
            plt.savefig(savedir +'/'+ str(it) + '.png')
        if show_in_console:
            plt.show()
        plt.close()

    def addtgt(self, tgt, fontsize= 14, offset=0.1, Chinese=False,Astro = False):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        assert 'boxes' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist() 
        numbox = tgt['boxes'].shape[0]

        color = []
        polygons = []
        boxes = []
        for box in tgt['boxes'].cpu():
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            color.append(c)

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=1)
        ax.add_collection(p)

        bbox_y_s = [bbox[1] for bbox in boxes]
        max_y = max(bbox_y_s)
        if Chinese:
            font_prop = FontProperties(fname='/home/rbaena/NotoSansCJK-Regular.ttc', size=fontsize)
        if Astro:
            font_prop = FontProperties(fname='/home/rbaena/NotoSansSymbols-VariableFont_wght.ttf', size=fontsize)
        if 'box_label' in tgt:
            assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
            for idx, bl in enumerate(tgt['box_label']):
                _string = str(bl)
                bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
                if Chinese:
                    ax.text(bbox_x, max_y+offset, _string, fontsize=fontsize, color=color[idx], bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1,'edgecolor': 'none'}, fontproperties=font_prop)
                elif Astro:
                    ax.text(bbox_x, max_y+offset, _string, fontsize=fontsize, color=color[idx], bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1}, fontproperties=font_prop)
                else:
                    ax.text(bbox_x, max_y+offset, _string, fontsize=fontsize, color=color[idx], bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1,'edgecolor': 'none'})
                if _string.strip() == "":
                    _string = "\u200b" 
                    ax.text(bbox_x, max_y+offset, _string, fontsize=fontsize, color=color[idx], bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1,'edgecolor': 'none'})
                    # rect  =  mpatches.Rectangle((bbox_x, max_y+offset), bbox_w, bbox_h, fill=True, facecolor = color[idx], alpha=0.6, edgecolor = 'none')
                    # ax.add_patch(rect)
                    
        if 'caption' in tgt:
            ax.set_title(tgt['caption'], wrap=True)


