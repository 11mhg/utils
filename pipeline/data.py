from __future__ import print_function

import os, time, json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .bbox import Box 
from .read_tfrecord_utils import *
from .write_tfrecord_utils import *

class DataReader:
    def __init__(self,classification=False,batch_size=32, shuffle_buffer_size=4, prefetch_buffer_size=1,
            num_parallel_calls=4,
            num_parallel_readers=1,image_size=(416,416),sequence=False):
        self.classification = classification
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.num_parallel_calls = num_parallel_calls
        self.num_parallel_readers = num_parallel_readers
        self.image_size = image_size
        self.sequence = sequence

    def get_batch(self,filenames=None):
        return input_fn(self,filenames)


class DataWriter:
    def __init__(self,inp_dir,out_dir,num_elem_per_shard):
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.num_elem_per_shard = num_elem_per_shard
        self.max_boxes = 0

    def convert_to(self):
        convert_to(self)


    def coco(self,dataset='train',year='2014'):
        start = time.time()
        if not os.path.exists(os.path.join(self.inp_dir,'images/'+dataset+year)):
            raise ValueError("Either Dataset or year is wrong, please choose a valid year or dataset!")
        self.name = dataset+year
        img_dir = os.path.join(self.inp_dir,'images/'+self.name+'/')
        annot_file = os.path.join(self.inp_dir,'annotations/instances_{}.json'.format(self.name))
        if not os.path.exists(annot_file): 
            raise ValueError("No annotation found at : {}".format(annot_file))
        with open(annot_file,'r') as f:
            annot_json = json.load(f)

        print("Beginning processing of {}.".format(self.name))
        out_dict = {}
        for i in range(len(annot_json['images'])):
            image_name = os.path.join(img_dir,annot_json['images'][i]['file_name'])
            if not os.path.exists(image_name):
                continue
            img_id = annot_json['images'][i]['id']
            out_dict[img_id] = {}
            out_dict[img_id]['filename'] = image_name
            out_dict[img_id]['boxes'] = []
            out_dict[img_id]['labels']= []
            out_dict[img_id]['size'] = [float(annot_json['images'][i]['width']),float(annot_json['images'][i]['height'])]
        
        pbar = tqdm(range(len(annot_json['annotations'])))
        pbar.set_description("Loading in COCO annotations and image paths")
        for i in pbar:
            img_id = annot_json['annotations'][i]['image_id']
            
            #bbox formatting
            bbox = [float(b) for b in annot_json['annotations'][i]['bbox']]
            xmin = bbox[0]
            ymin = bbox[1]
            b_w = bbox[2]
            b_h = bbox[3]
            xmax = xmin+b_w
            ymax = ymin+b_h
            xmin = xmin/out_dict[img_id]['size'][0]
            xmax = xmax/out_dict[img_id]['size'][0]
            ymin = ymin/out_dict[img_id]['size'][1]
            ymax = ymax/out_dict[img_id]['size'][1]

            bbox = [xmin,ymin,xmax,ymax]
            label = float(annot_json['annotations'][i]['category_id'])

            out_dict[img_id]['boxes'].append(bbox)
            out_dict[img_id]['labels'].append(label)

        print("Number of images loaded in: {}".format(len(out_dict.keys())))
        print("Number of boxes loaded in: {}".format(len(annot_json['annotations'])))

        self.images = []
        self.labels = []

        pbar = tqdm(out_dict.keys())
        pbar.set_description("Converting to internal Format")
        for img_id in pbar:
            image_path = out_dict[img_id]['filename']
            bxs = out_dict[img_id]['boxes']
            ls = out_dict[img_id]['labels']
            boxes = []
            for ind in range(len(bxs)):
                boxes.append(Box(x0=bxs[ind][0],y0=bxs[ind][1],x1=bxs[ind][2],y1=bxs[ind][3],label=ls[ind]))
            if len(boxes) > self.max_boxes:
                self.max_boxes = len(boxes)
            self.images.append(image_path)
            self.labels.append(np.array(boxes))
        print("Finished Loading In COCO: {}".format(self.name))
        end = time.time()
        print("Total Time taken: {:.4f} seconds".format(end-start))
        assert len(self.images) == len(self.labels)
        self.num_shards = len(self.images)//self.num_elem_per_shard
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        return

    def MOT(self,dataset='train'):
        start = time.time()
        self.name = dataset
        dirs = os.path.join(self.inp_dir,dataset)
        scenes = [dirs+'/'+i+'/' for i in os.listdir(dirs)]
        
        self.images = []
        self.labels = []
        pbar = tqdm(scenes)
        
        for scene_dir in pbar:
            img_locs = scene_dir+'img1/'
            annot_loc = scene_dir+'gt/gt.txt'

            with open(scene_dir+'seqinfo.ini','r') as info:
                for lines in info:
                    if 'imWidth' in lines:
                        lines = lines.split('=')
                        width = float(lines[1])
                    elif 'imHeight' in lines:
                        lines = lines.split('=')
                        height = float(lines[1])
                    else:
                        continue
            assert height!=0
            assert width!=0

            with open(annot_loc,'r') as gt_file:
                dict_annot = {}
                dict_annot['frame'] = {}
                for index, lines in enumerate(gt_file):
                    splitline = [float(x.strip()) for x in lines.split(',')]
                    label = int(splitline[7])-1
                    x_val = splitline[2]/width
                    y_val = splitline[3]/height
                    box_width = splitline[4]/width
                    box_height = splitline[5]/height

                    x0 = x_val
                    y0 = y_val
                    x1 = x_val+box_width
                    y1 = y_val+box_height
                    box = Box(x0=x0,y0=y0,x1=x1,y1=y1,label=label)
                    frame_id = int(splitline[0])
                    if frame_id not in dict_annot['frame']:
                        dict_annot['frame'][frame_id] = []
                    dict_annot['frame'][frame_id].append(box)
            for frame_id in sorted(dict_annot['frame'].keys()):
                pbar.set_description('Sorting through frame {} of scene'.format(frame_id))
                boxes = np.array(dict_annot['frame'][frame_id])
                if len(boxes) > self.max_boxes:
                    self.max_boxes = len(boxes)
                img_name = img_locs + str(frame_id).zfill(6)+'.jpg'
                self.images.append(img_name)
                self.labels.append(boxes)
        print("Finished Loading In MOT")
        end = time.time()
        print("Total Time taken: {:.4f} seconds".format(end-start))
        assert len(self.images) == len(self.labels)
        self.num_shards = len(self.images)//self.num_elem_per_shard
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        return

