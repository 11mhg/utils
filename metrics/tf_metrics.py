import tensorflow as tf

#all the following calculations come with the assumption
#That the pairwise boxes are already matched
#Which is to say that the closes box is already chosen.
#Also assumes x0y0x1y1
def tf_iou_corners(bboxes1,bboxes2,scope=None):
    with tf.name_scope(scope,'IOU'):
        x11, y11, x12, y12 = tf.split(bboxes1,4,axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2,4,axis=1)

        xI1 = tf.maximum(x11,tf.transpose(x21))
        yI1 = tf.maximum(y11,tf.transpose(y21))
    
        xI2 = tf.minimum(x12,tf.transpose(x22))
        yI2 = tf.minimum(y12,tf.transpose(y22))
    
        inter_area = tf.maximum((xI2 - xI1),0) * tf.maximum((yI2 - yI1),0)
    
        bboxes1_area = (x12 - x11) * (y12 - y11)
        bboxes2_area = (x22 - x21) * (y22 - y21)
    
        union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area
    
        epsilon = tf.constant(0.0001,dtype=tf.float32)
    
        return inter_area / (union+epsilon)

#Returns a generalized IOU 
#Basic idea is that instead of flattening to zero
#We return a value between -1 and 1
def tf_giou_corners(bboxes1, bboxes2, scope=None):
    with tf.name_scope(scope,'g_IOU'):
        x11, y11, x12, y12 = tf.split(bboxes1,4,axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2,4,axis=1)

        xI1 = tf.maximum(x11,tf.transpose(x21))
        yI1 = tf.maximum(y11,tf.transpose(y21))
    
        xI2 = tf.minimum(x12,tf.transpose(x22))
        yI2 = tf.minimum(y12,tf.transpose(y22))
    
        x1c = tf.minimum(x11,tf.transpose(x21))
        y1c = tf.minimum(y11,tf.transpose(y21))
        x2c = tf.maximum(x12,tf.transpose(x22))
        y2c = tf.maximum(y12,tf.transpose(y22))

        enclose_area = tf.maximum((x2c-x1c),0)*tf.maximum((y2c-y1c),0)

        inter_area = tf.maximum((xI2 - xI1),0) * tf.maximum((yI2 - yI1),0)
    
        bboxes1_area = (x12 - x11) * (y12 - y11)
        bboxes2_area = (x22 - x21) * (y22 - y21)
    
        union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area
    
        epsilon = tf.constant(0.0001,dtype=tf.float32)
        IOU = inter_area / (union+epsilon)
        
        GIOU = IOU - ((enclose_area - union)/(enclose_area))
        return GIOU

def tf_iou_centers(bboxes1,bboxes2,scope=None):
    with tf.name_scope(scope,'IOU_centers') as new_scope:
        bboxes1 = tf_x0y0(bboxes1,scope=new_scope)
        bboxes2 = tf_x0y0(bboxes2,scope=new_scope)
        return tf_iou_corners(bboxes1,bboxes2,scope=new_scope)

def tf_giou_centers(bboxes1,bboxes2,scope=None):
    with tf.name_scope(scope,'gIOU_centers') as new_scope:
        bboxes1 = tf_x0y0(bboxes1,scope=new_scope)
        bboxes2 = tf_x0y0(bboxes2,scope=new_scope)
        return tf_giou_corners(bboxes1,bboxes2,scope=new_scope)

#Converst cxcy to x0y0
def tf_x0y0(bboxes,scope=None):
    with tf.name_scope(scope,'cxcy_to_x0y0'):
        cx, cy, w, h = tf.split(bboxes,4,axis=1)

        x0 = cx - (w/2.)
        x1 = cx + (w/2.)
        y0 = cy - (h/2.)
        y1 = cy + (h/2.)

        return tf.concat([x0,y0,x1,y1],axis=-1)

def tf_cxcy(bboxes,scope=None):
    with tf.name_scope(scope,'x0y0_to_cxcy'):
        x0,y0,x1,y1 = tf.split(bboxes,4,axis=1)

        w = x1-x0
        h = y1-y0
        cx = x0+(w/2.)
        cy = y0+(h/2.)

        return tf.concat([cx,cy,w,h],axis=-1)
