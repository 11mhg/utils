import tensorflow as tf
import numpy as np
from .tf_metrics import *

def x0y0tocxcy(box):
    x0, y0, x1, y1 = box[0], box[1],box[2],box[3]
    w = x1 - x0
    h = y1 - y0
    cx = x0 +(w/2.)
    cy = y0 +(h/2.)

    return [cx,cy,w,h]

def cxcytox0y0(box):
    cx, cy, w, h = box[0],box[1],box[2],box[3]
    x0 = cx-(w/2.)
    y0 = cy-(h/2.)
    x1 = x0+w
    y1 = y0+h
    return [x0,y0,x1,y1]


def test_boxes():
    bboxes1 = tf.placeholder(tf.float32,[None,4])
 
    bboxes1_vals = [[39.,63.,203.,112.],[0.,0.,10.,10.]]
    
    bboxes_1 = [x0y0tocxcy(i) for i in bboxes1_vals]

    t_boxes1 = tf_x0y0(bboxes1)
    with tf.Session() as sess:
        new_boxes1 = sess.run(t_boxes1,
                feed_dict={bboxes1:bboxes_1})

    print("Acutal values (x0,y0,x1,y1):")
    print(bboxes1_vals)
    print("cxcy")
    print(bboxes_1)
    print("Recreated values:")
    print(new_boxes1)






def test_tf_iou_corners():
    bboxes1 = tf.placeholder(tf.float32,[None,4])
    bboxes2 = tf.placeholder(tf.float32,[None,4])

    bboxes1_vals = [[39.,63.,203.,112.],[0.,0.,10.,10.]]
    
    bboxes2_vals = [[3.,4.,25.,32.],[54.,66.,198.,114.],[6.,7.,60.,44.]]
    
    overlap_op = tf_iou_corners(bboxes1,bboxes2)
    with tf.Session() as sess:
        overlap = sess.run(overlap_op, feed_dict={
            bboxes1: np.array(bboxes1_vals),
            bboxes2: np.array(bboxes2_vals),
        })
        print(overlap)
        print(np.amax(overlap,axis=-1))


def test_tf_iou_centers():
    bboxes1 = tf.placeholder(tf.float32,[None,4])
    bboxes2 = tf.placeholder(tf.float32,[None,4])

    overlap_op = tf_iou_centers(bboxes1,bboxes2)

    bboxes1_vals = [[39.,63.,203.,112.],[0.,0.,10.,10.]]
    
    bboxes2_vals = [[3.,4.,25.,32.],[54.,66.,198.,114.],[6.,7.,60.,44.]]   

    bboxes1_vals = [x0y0tocxcy(i) for i in bboxes1_vals]
    bboxes2_vals = [x0y0tocxcy(i) for i in bboxes2_vals]

    with tf.Session() as sess:
        overlap = sess.run(overlap_op, feed_dict={
            bboxes1: np.array(bboxes1_vals),
            bboxes2: np.array(bboxes2_vals),
        })
        print(overlap)
        print(np.amax(overlap,axis=-1))




def test_tf_giou_corners():
    bboxes1 = tf.placeholder(tf.float32,[None,4])
    bboxes2 = tf.placeholder(tf.float32,[None,4])

    bboxes1_vals = [[39.,63.,203.,112.],[0.,0.,10.,10.]]
    
    bboxes2_vals = [[3.,4.,25.,32.],[54.,66.,198.,114.],[6.,7.,60.,44.]]
    
    overlap_op = tf_giou_corners(bboxes1,bboxes2)
    with tf.Session() as sess:
        overlap = sess.run(overlap_op, feed_dict={
            bboxes1: np.array(bboxes1_vals),
            bboxes2: np.array(bboxes2_vals),
        })
        print(overlap)
        print(np.amax(overlap,axis=-1))


def test_tf_giou_centers():
    bboxes1 = tf.placeholder(tf.float32,[None,4])
    bboxes2 = tf.placeholder(tf.float32,[None,4])

    overlap_op = tf_giou_centers(bboxes1,bboxes2)

    bboxes1_vals = [[39.,63.,203.,112.],[0.,0.,10.,10.]]
    
    bboxes2_vals = [[3.,4.,25.,32.],[54.,66.,198.,114.],[6.,7.,60.,44.]]   

    bboxes1_vals = [x0y0tocxcy(i) for i in bboxes1_vals]
    bboxes2_vals = [x0y0tocxcy(i) for i in bboxes2_vals]

    with tf.Session() as sess:
        overlap = sess.run(overlap_op, feed_dict={
            bboxes1: np.array(bboxes1_vals),
            bboxes2: np.array(bboxes2_vals),
        })
        print(overlap)
        print(np.amax(overlap,axis=-1))


def test_all():
    print("Beginning test of all metrics in tf_metrics")
    test_boxes()
    test_tf_iou_centers()
    test_tf_iou_corners()
    test_tf_giou_centers()
    test_tf_giou_corners()

