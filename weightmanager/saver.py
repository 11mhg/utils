import tensorflow as tf
import numpy as np
import time, os, random


class Saver():
    def __init__(self,ckpt_dir,pretrain_var=None,pretrain_dir=None):
        self.pretrain_vars = pretrain_var
        self.pretrain_dir = pretrain_dir
        self.ckpt_dir = ckpt_dir

        if self.pretrain_vars is not None:
            self.pretrain_saver = tf.train.Saver(self.pretrain_vars)
        
        self.vars = [v for v in tf.trainable_variables()]
        self.saver = tf.train.Saver(self.vars)

    def restore_pretrain(self,sess):
        try:
            self.pretrain_saver.restore(sess,tf.train.latest_checkpoint(self.pretrain_dir))
            tf.logging.info("Pretrain Saver succeeded in restoring variables")
        except:
            tf.logging.info("Pretrain Saver did not succeed in restoring variables")

    def restore(self,sess):
        try:
            self.saver.restore(sess,tf.train.latest_checkpoint(self.ckpt_dir))
            tf.logging.info("Saver succeeded in restoring variables")
        except:
            tf.logging.info("Saver did not succeed in restoring variables")

    def save_pretrain(self,sess,global_step=None):
        try:
            self.pretrain_saver.save(sess,self.pretrain_dir,global_step=global_step)
            tf.logging.info("Pretrain Saver succeeded in saving variables")
        except:
            tf.logging.info("Pretrain Saver did not succeed in saving variables")

    def save(self,sess,global_step=None):
        try:
            self.saver.save(sess,self.ckpt_dir,global_step=global_step)
            tf.logging.info("Saver succeeded in saving variables")
        except:
            tf.logging.info("Saver did not succeed in saving variables") 
