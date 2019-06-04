import numpy as np
import tensorflow as tf
import os, math, cv2, random


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


def input_fn(self, filenames):
    files = tf.data.Dataset.list_files(filenames)
    #read files in parallel and decrypt in parallel
    with tf.device('/cpu:0'):
        dataset = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset,cycle_length = self.num_parallel_readers))

        dataset = dataset.shuffle(buffer_size= self.shuffle_buffer_size)

        if self.classification:
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                map_func = class_parse_fn, batch_size=self.batch_size))
        else:
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                map_func = box_parse_fn, batch_size=self.batch_size))
        dataset = dataset.prefetch(buffer_size = self.prefetch_buffer_size)
    return dataset

def box_parse_fn(example):
    #parse tfexample record
    example_fmt = {
            'image/image_raw': tf.FixedLenFeature([],tf.string,""),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/labels': tf.VarLenFeature(dtype=tf.float32),
            'image/object/num_objects': tf.FixedLenFeature([],dtype=tf.int64),
            'image/object/max_boxes': tf.FixedLenFeature([],dtype=tf.int64),
    }
    parsed = tf.parse_single_example(example,example_fmt)
    num_objects = tf.cast(parsed['image/object/num_objects'],tf.int32)
    max_objects = tf.cast(parsed['image/object/max_boxes'],tf.int32)

    x0 = tf.sparse_tensor_to_dense(parsed['image/object/bbox/xmin'],default_value=0)
    y0 = tf.sparse_tensor_to_dense(parsed['image/object/bbox/ymin'],default_value=0)
    x1 = tf.sparse_tensor_to_dense(parsed['image/object/bbox/xmax'],default_value=0)
    y1 = tf.sparse_tensor_to_dense(parsed['image/object/bbox/ymax'],default_value=0)
    labels = tf.sparse_tensor_to_dense(parsed['image/object/bbox/labels'],default_value=0)

    b_shape = tf.stack([max_objects,1])

    x0 = tf.reshape(x0,b_shape)
    x1 = tf.reshape(x1,b_shape)
    y0 = tf.reshape(y0,b_shape)
    y1 = tf.reshape(y1,b_shape)
    labels = tf.reshape(labels,b_shape)

    b_w = x1 - x0
    b_h = y1 - y0

    cx = x0 + (b_w/2.)
    cy = y0 + (b_h/2.)

    boxes = tf.concat([cx,cy,b_w,b_h],axis=-1)
    #decode image
    image = tf.image.decode_jpeg(parsed['image/image_raw'],channels=3)
    image_shape = tf.shape(image)
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.image.resize_images(image,[416,416],align_corners=True)
    max_val = tf.reduce_max(image)

    image = tf.cond(max_val > 1.0, lambda: image/255.0, lambda: tf.identity(image)) 
    return image,labels, boxes, num_objects

def class_parse_fn(example):
    #parse tfexample record
    example_fmt = {
            'image/image_raw': tf.FixedLenFeature([],tf.string,""),
            'image/object/bbox/labels': tf.VarLenFeature(dtype=tf.float32),
    }
    parsed = tf.parse_single_example(example,example_fmt)

    label = tf.expand_dims(parsed['image/object/bbox/labels'].values,0)

    #decode image
    image = tf.image.decode_jpeg(parsed['image/image_raw'],channels=3)
    image_shape = tf.shape(image)
    image = tf.image.convert_image_dtype(image,tf.float32)
    max_val = tf.reduce_max(image)

    image = tf.cond(max_val > 1.0, lambda: image/255.0, lambda: tf.identity(image))
    return image, label


