import numpy as np
import tensorflow as tf
import os, math, cv2, random, time
import multiprocessing
from multiprocessing import Process

def _int64_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floats_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _process_image_files(imgs,labels,records,cpu_num,max_boxes):
    print("{} beginning processing of records.".format(cpu_num))
    count = 0
    for ind in range(len(records)):
        try:
            images = imgs[ind]
        except:
            print(ind)
            print(cpu_num,len(records),imgs.shape)
        filename = records[ind]
        l = labels[ind]
        with tf.python_io.TFRecordWriter(filename) as writer:
            count +=1 
            if cpu_num == 0:
                print("%.4f done in proc 0 percent."%(count / imgs.size))
            for index in range(images.shape[0]):
                image_name = images[index]
                try:
                    image = open(image_name,'rb').read()
                except:
                    continue
                xmin,ymin,xmax,ymax = [],[],[],[]
    
                label = []
    
                #Get XYXY components of box
                boxes = l[index]
   
                for i in range(max_boxes):
                    if i < (boxes.shape[0]):
                        box = boxes[i]
                        b = box.xyxy
                        xmin.append(b[0])
                        ymin.append(b[1])
                        xmax.append(b[2])
                        ymax.append(b[3])
                        label.append(box.label)
                    else:
                        xmin.append(0)
                        ymin.append(0)
                        xmax.append(0)
                        ymax.append(0)
                        label.append(-1)
                assert (len(xmin)==len(ymin)==len(xmax)==len(ymax)==len(label)) 
                num_boxes = [boxes.shape[0]]
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image/object/bbox/xmin': _floats_feature(xmin),
                            'image/object/bbox/ymin': _floats_feature(ymin),
                            'image/object/bbox/xmax': _floats_feature(xmax),
                            'image/object/bbox/ymax': _floats_feature(ymax),
                            'image/object/bbox/labels': _floats_feature(label),
                            'image/object/num_objects': _int64_feature(num_boxes),
                            'image/object/max_boxes': _int64_feature(max_boxes),
                            'image/image_raw': _bytes_feature(tf.compat.as_bytes(image)),
                        }))
                writer.write(example.SerializeToString())
    print("{} Process done processing.".format(cpu_num))
    return




def convert_to(self):
    start_time = time.time()
    if not os.path.exists(self.out_dir):
        os.mkdir(self.out_dir)
    
    filenames = [os.path.join(self.out_dir,self.name+'_{}'.format(i)+'.tfrecords') for i in range(self.num_shards)]
#    ind_arr = np.arange(len(filenames))
    
    cpu_count = os.cpu_count()

    split = np.array_split(np.array(filenames),max(1,cpu_count-1))
#    for i in split:
#        print(i.shape)

    ind_img = np.arange(len(self.images))
    
    img_split = np.array_split(np.array(ind_img),self.num_shards)
    img_split = np.array_split(np.array(img_split),max(1,cpu_count-1))
#    print(np.array(img_split).shape)
#    print(np.array(img_split[0]).shape)
#    print(np.array(img_split[0][0]).shape)
#    print(np.array(img_split[0][0][0]).shape)
    coord = tf.train.Coordinator()
    processes = []

    for i in range(cpu_count-1):
        cur_images_ind = img_split[i]
        cur_records = split[i]
        cur_images = []
        cur_labels = []
        for shard_ind in cur_images_ind:
            cur_images.append([])
            cur_labels.append([])
            cur_images[-1] = self.images[shard_ind]
            cur_labels[-1] = self.labels[shard_ind]

        cur_images = np.array(cur_images)
        cur_labels = np.array(cur_labels)
        cur_records = np.array(cur_records)
        args = (cur_images,cur_labels,cur_records,i,self.max_boxes)
        p = Process(target=_process_image_files,args = args)
        p.start()
        processes.append(p)
    print("All Processes have been started.")
    coord.join(processes)
    end_time = time.time()
    print("Completed writing all records")
    print("Process took %.4f seconds."%(end_time-start_time))



















