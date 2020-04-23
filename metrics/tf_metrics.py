import sys
from cmath import cos, sin, pi
import numpy as np
import tensorflow as tf


#Batched versions of code

'''
Processes a batch of ious assuming x1y1x2y2
'''
def tf_batched_iou_corners(bboxes1,bboxes2,scope=None):
    with tf.name_scope(scope,'batched_IOU') as new_scope:
        @tf.function
        def func(bboxes1,bboxes2,scope=None):
            bs = bboxes.shape[0]
            batch_ious = tf.TensorArray(tf.float32,size=bs)
            for i in tf.range(bs):
                boxes1 = bboxes1[i]
                boxes2 = bboxes2[i]
                iou = tf_iou_corners(boxes1,boxes2,scope=new_scope)
                batch_ious = batch_ious.write(i,iou)
            return batch_ious
        return func(bboxes1,bboxes2,new_scope)

'''
Same as above but for giou
'''
def tf_batched_giou_corners(bboxes1,bboxes2,scope=None):
    with tf.name_scope(scope,'batched_GIOU') as new_scope:
        @tf.function
        def func(bboxes1,bboxes2,scope=None):
            bs = bboxes.shape[0]
            batch_gious = tf.TensorArray(tf.float32,size=bs)
            for i in tf.range(bs):
                boxes1 = bboxes1[i]
                boxes2 = bboxes2[i]
                iou, giou = tf_giou_corners(boxes1,boxes2,scope=new_scope)
                batch_gious = batch_gious.write(i,giou)
            return batch_gious
        return func(bboxes1,bboxes2,new_scope)



#all the following calculations come with the assumption
#you are comparing all boxes with all other boxes
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
    
        iou =  inter_area / (union+epsilon)

        #If we have a NaN, we divided by zero (this should not happen, but should not is different
        #than does not. Instead, replace with worst possible value of 0
#        iou = tf.where(tf.is_nan(iou),tf.zeros_like(iou),iou)
        return iou


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

        inter_area = tf.maximum(tf.maximum((xI2 - xI1),0) * tf.maximum((yI2 - yI1),0),0)
    
        bboxes1_area = (x12 - x11) * (y12 - y11)
        bboxes2_area = (x22 - x21) * (y22 - y21)
    
        union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area
    
        epsilon = tf.constant(0.0001,dtype=tf.float32)
        IOU = inter_area / (union+epsilon)
        
        GIOU = tf.minimum(tf.maximum(IOU - ((enclose_area - union)/(enclose_area+epsilon)),-1),1)
        
        #If we have a nan, it is because we are dividing by 0, just return
        #worst possible GIOU  

        
#        IOU = tf.where(tf.is_nan(IOU),tf.zeros_like(IOU),IOU)

#        GIOU = tf.where(tf.is_nan(GIOU),-tf.ones_like(GIOU),GIOU) 
        return IOU, GIOU

#Just a convenience function that turns centers to corners and then return
#the iou of those two.
def tf_iou_centers(bboxes1,bboxes2,scope=None):
    with tf.name_scope(scope,'IOU_centers') as new_scope:
        bboxes1 = tf_x0y0(bboxes1,scope=new_scope)
        bboxes2 = tf_x0y0(bboxes2,scope=new_scope)
        return tf_iou_corners(bboxes1,bboxes2,scope=new_scope)

#Same as above but for GIOU
def tf_giou_centers(bboxes1,bboxes2,scope=None):
    with tf.name_scope(scope,'gIOU_centers') as new_scope:
        bboxes1 = tf_x0y0(bboxes1,scope=new_scope)
        bboxes2 = tf_x0y0(bboxes2,scope=new_scope)
        return tf_giou_corners(bboxes1,bboxes2,scope=new_scope)

#Convert cxcy to x0y0 (remember boxes are assumed to be [num_boxes,4])
def tf_x0y0(bboxes,scope=None):
    with tf.name_scope(scope,'cxcy_to_x0y0'):
        cx, cy, w, h = tf.split(bboxes,4,axis=1)

        x0 = cx - (w/2.)
        x1 = cx + (w/2.)
        y0 = cy - (h/2.)
        y1 = cy + (h/2.)

        return tf.concat([x0,y0,x1,y1],axis=-1)

#convert x0y0x1y1 to cxcy
def tf_cxcy(bboxes,scope=None):
    with tf.name_scope(scope,'x0y0_to_cxcy'):
        x0,y0,x1,y1 = tf.split(bboxes,4,axis=1)

        w = x1-x0
        h = y1-y0
        cx = x0+(w/2.)
        cy = y0+(h/2.)

        return tf.concat([cx,cy,w,h],axis=-1)

#convenience function for diff 
def tf_diff(tensor):
    return tensor[1:]-tensor[:-1]


#performs the tf ldlj iou on temporal boxes of shape assuming boxes are in x0y0x1y1
#[TIMESTEPS,4]
def tf_iou_ldlj(boxes,fs,scope=None):
    with tf.name_scope(scope,'iou_ldlj') as new_scope:
        ious = tf_iou_corners(boxes[:-1],boxes[1:],scope = new_scope)
        diag = tf.linalg.diag_part(ious)
        s_ious = 1. - diag
        s_ious = tf.debugging.check_numerics(s_ious, 's_ious in ldlj has NaN')
        ldlj_val = _ldlj(s_ious,fs,scope=new_scope)
        ldlj_val = tf.debugging.check_numerics(ldlj_val, '_ldlj produced a NaN')
    return ldlj_val


#Sub function for calculation LDLJ
def _ldlj(v, fs, peak = 1, scope=None):
    with tf.name_scope(scope, 'ldlj') as new_scope:
        a = tf_diff(v) #acceleration is diff of velocity
        j = tf_diff(a) #jerk is diff of acceleration
        peak = tf.constant(peak,dtype=tf.float32)
        #tf.reduce_max(v) + tf.constant(1e-8,tf.float32)
        dt = tf.constant(1. /fs, tf.float32)
        duration = tf.cast(tf.size(v),tf.float32) * dt
        scale = tf.truediv(tf.pow(duration,5),tf.pow(peak,2))
        dlj_val = scale * tf.reduce_sum(tf.pow(j,2)) * dt
        ldlj_val = - tf.log1p(dlj_val)
    return ldlj_val

#As tf_giou_ldlj but for gioud as base speed
def tf_giou_ldlj(boxes,fs,scope=None):
    with tf.name_scope(scope,'giou_ldlj') as new_scope:
        _, gious = tf_giou_corners(boxes[:-1],boxes[1:],scope = new_scope)
        s_gious = 1. - tf.linalg.diag_part(gious)
        ldlj_val = _ldlj(s_gious, fs, peak = 2., scope=new_scope)
    return ldlj_val


#performs the SAL on the IOU with particular windowsizes and amp/fc thresholding
def tf_iou_sal(boxes, fs, window_size=16, padlevel=4, fc=10.0, amp_th= 0.001, reduce_sum=True, scope=None): 
    with tf.name_scope(scope,'iou_sal') as new_scope:
        fs = tf.constant(fs,dtype=tf.float32)
        fc = tf.constant(fc,dtype=tf.float32)
        amp_th = tf.constant(amp_th,dtype=tf.float32)

        ious = tf_iou_corners(boxes[:-1],boxes[1:],scope=new_scope)
        s_ious = 1. - tf.linalg.diag_part(ious)
        s_ious = s_ious
        f =tf.range(tf.constant(0),fs,
                fs/tf.constant(window_size-1,dtype=tf.float32))
#        print(s_ious)
        sdft_obj = tf_sdft(N=window_size-1,scope=new_scope)
        freqs = sdft_obj.sdft_func(s_ious,scope=new_scope)
        freqs = tf.real(freqs)
        Mf = tf.abs(freqs)

        #Normalize???
        #max_Mf = tf.reduce_max(Mf)
        #Mf = tf.truediv(Mf,max_Mf)

        fc_inx = tf.math.less_equal(f,fc)
        f_sel = tf.boolean_mask(f, fc_inx) 
        Mf_sel = tf.boolean_mask(Mf, fc_inx)

        inx = tf.math.greater_equal(Mf_sel,amp_th)
        f_sel = tf.boolean_mask(f_sel,inx)
        Mf_sel = tf.boolean_mask(Mf_sel,inx)



        #Certain cases give us an f_sel and Mf_sel that are far too noisy and
        #don't register properly. Give the worst possible SAL for this.
        f_sel_num = tf.size(f_sel)

        def empty_fsel():
            freq = tf.constant([0,1],dtype=tf.float32,name='template-f_sel')
            Mfreq = tf.constant([1,0.05],dtype=tf.float32,name='template-Mf_sel')
            print_op = tf.print("Substituted by: ",freq,Mfreq)
            with tf.control_dependencies([print_op]):
                freq = tf.identity(freq)
                Mfreq = tf.identity(Mfreq)
            return freq, Mfreq

        def not_empty_fsel():
            freq = tf.identity(f_sel,name='not_empty_fsel')
            Mfreq = tf.identity(Mf_sel,name='not_empty_mfsel')
            return freq, Mfreq

        cond_bool = tf.math.equal(f_sel_num, tf.constant(0))
        f_sel, Mf_sel = tf.cond( cond_bool, empty_fsel , not_empty_fsel )
        
        new_sal = tf.sqrt(tf.pow(tf_diff(f_sel)/(tf.abs(f_sel[-1]-f_sel[0])), 2.) + tf.pow(tf_diff(Mf_sel), 2.)) 
        if reduce_sum:
            new_sal = -tf.reduce_sum(new_sal)
        else:
            new_sal = -1. * new_sal
        new_sal = tf.real(new_sal)
        new_sal = tf.cast(new_sal,dtype=tf.float32)
        return new_sal



def tf_giou_sal(boxes, fs, window_size=16, padlevel=1, fc=10.0, amp_th= 0.001, scope=None):
    with tf.name_scope(scope,'giou_sal') as new_scope:
        fs = tf.constant(fs,dtype=tf.float32)
        fc = tf.constant(fc,dtype=tf.float32)
        amp_th = tf.constant(amp_th,dtype=tf.float32)
        
        _,gious = tf_giou_corners(boxes[:-1],boxes[1:],scope=new_scope)
        s_ious = 1. - tf.linalg.diag_part(gious)

        f = tf.range(tf.constant(0),fs,
                fs/tf.constant(window_size-1,dtype=tf.float32))
        
        sdft_obj = tf_sdft(N=window_size-1,scope=new_scope)
        freqs = sdft_obj.sdft_func(s_ious,scope=new_scope)

        Mf = tf.abs(freqs)

        #Normalize???
        max_Mf = tf.reduce_max(Mf)
        Mf = tf.truediv(Mf,max_Mf)

        fc_inx = tf.math.less_equal(f,fc)
        f_sel = tf.boolean_mask(f, fc_inx) 
        Mf_sel = tf.boolean_mask(Mf, fc_inx)

        inx = tf.math.greater_equal(Mf_sel,amp_th)
        f_sel = tf.boolean_mask(f_sel,inx)
        Mf_sel = tf.boolean_mask(Mf_sel,inx)

        #Certain cases give us an f_sel and Mf_sel that are far too noisy and
        #don't register properly. Give the worst possible SAL for this.
        f_sel_num = tf.size(f_sel)

        def empty_fsel():
            freq = tf.constant([0,1],dtype=tf.float32,name='template-f_sel')
            Mfreq = tf.constant([1,0.05],dtype=tf.float32,name='template-Mf_sel')
            print_op = tf.print("Substituted by: ",freq,Mfreq)
            with tf.control_dependencies([print_op]):
                freq = tf.identity(freq)
                Mfreq = tf.identity(Mfreq)
            return freq, Mfreq

        def not_empty_fsel():
            freq = tf.identity(f_sel)
            Mfreq = tf.identity(Mf_sel)
            return freq, Mfreq

        cond_bool = tf.math.equal(f_sel_num, tf.constant(0))
        f_sel, Mf_sel = tf.cond( cond_bool, empty_fsel , not_empty_fsel )


#        print_op = tf.print("Output of f_sel: ",f_sel,"\nOutput of Mf_sel:"
#                ,Mf_sel, "\nf_sel_num: ",f_sel_num, "\nboolean condition:"
#                ,cond_bool)
#        with tf.control_dependencies([print_op]):
        new_sal = -tf.reduce_sum(tf.sqrt(tf.pow(tf_diff(f_sel)/(f_sel[-1]-f_sel[0]), 2.) +
            tf.pow(tf_diff(Mf_sel), 2.)))

        return new_sal




class tf_sdft:

    def __init__(self,N=16,scope=None):
        self.N = N
        self.scope = scope
        self.initialize()

    #set initial values for twiddle and frequencies
    #and input signal
    def initialize(self):
        with tf.name_scope(self.scope,'sliding_dft') as new_scope:
            coeffs = []
            freqs = []
            in_s = []
            for i in range(self.N):
                a = 2.0 * pi * i /self.N
                coeffs.append(complex(cos(a),sin(a)))
                freqs.append(complex(0,0))
                in_s.append(complex(0,0))
            coeffs = np.array(coeffs,dtype=np.complex64)
            freqs = np.array(freqs,dtype=np.complex64)
            in_s = np.array(in_s,dtype=np.complex64)
            self.N_t = tf.constant(self.N)
         
            with tf.variable_scope(self.scope+"sdft_vars",reuse=tf.AUTO_REUSE):
                #Convert to variables so we can actually perform 
                self.coeffs = tf.get_variable('coeffs',coeffs.shape,
                        trainable=False,dtype=tf.complex64,
                        initializer=tf.constant_initializer(coeffs))
                self.freqs = tf.get_variable('freqs',freqs.shape,
                        trainable=False,dtype=tf.complex64,
                        initializer=tf.constant_initializer(freqs))
                self.in_s = tf.get_variable('in_s',in_s.shape,
                        trainable=False,dtype=tf.complex64,
                        initializer=tf.constant_initializer(in_s))
                self.SDFT_initializer = tf.initializers.variables([self.coeffs,self.freqs,self.in_s])

    def get_variables(self):
        with tf.variable_scope(self.scope+"sdft_vars",reuse=True):        
            #Convert to variables so we can actually perform 
            self.coeffs = tf.get_variable('coeffs',dtype=tf.complex64)
            self.freqs = tf.get_variable('freqs',dtype=tf.complex64)
            self.in_s = tf.get_variable('in_s',dtype=tf.complex64)
        return self.coeffs, self.freqs, self.in_s

    def sdft_func(self,input_tensor,scope=None):
        with tf.name_scope(scope,'sliding_DFT_func') as new_scope:
            @tf.function(autograph=True, experimental_compile=False)
            def func(input_tensor,N_t,in_s,coeffs,freqs):
                in_s = tf.identity(in_s,name='identity_in_s')
                coeffs = tf.identity(coeffs,name='identity_coeffs')
                freqs = tf.identity(freqs,name='identity_freqs')
    
    
                for i in range(N_t):
                    last = in_s[self.N_t-1]
        
                    in_s = in_s[:-1]
                    new_val = tf.expand_dims(tf.complex(input_tensor[i],
                        tf.cast(0.0,dtype=tf.float32)),0)
                    
                    in_s = tf.concat([new_val,in_s],axis=0)
                    delta = in_s[0] - last
                    
                    freq_delta = freqs + delta
                    freqs = tf.math.multiply(freq_delta,coeffs)
    
                return freqs,in_s

            new_freqs, new_in_s = func(input_tensor,self.N_t,
                    self.in_s,self.coeffs,self.freqs)
            
            self.in_s = self.in_s.assign(new_in_s)
            self.freqs = self.freqs.assign(new_freqs)
    
            with tf.control_dependencies([self.in_s,self.freqs]):
                new_freqs = tf.real(tf.identity(new_freqs))

        return new_freqs























