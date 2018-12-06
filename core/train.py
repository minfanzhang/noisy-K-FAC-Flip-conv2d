from core.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

import os
import tensorflow as tf
import pickle as pickle

GRAD_CHECK_ROOT_DIR = './grad_checks_FC_KFAC'


class Trainer(BaseTrain):
    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.checkpoint_dir = './checkpoint_KFAC_FC_flip'
        self.model_name = 'model'

    def train(self):
        if self.config.reload_step > 0 :
            print('---->reloading ', self.config.reload_step)
            self.reload(self.config.reload_step, self.sess, self.model.saver)

        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch(cur_epoch)
            self.test_epoch()

    def train_epoch(self, cur_epoch):
        loss_list = []
        acc_list = []
        for itr, (x, y) in enumerate(tqdm(self.train_loader)):
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.n_particles: self.config.train_particles
            }
            '''if itr % 20 == 0:
                acc_test_list = []
                for (x1, y1) in self.test_loader:
                    feed_dict_test = {
                        self.model.inputs: x1,
                        self.model.targets: y1,
                        self.model.is_training: False,
                        self.model.n_particles: self.config.test_particles
                    }
                    acc_test = self.sess.run([self.model.acc], feed_dict=feed_dict_test)

                    acc_test_list.append(acc_test)                

                avg_test_acc = np.mean(acc_test_list)
                self.logger.info("itr %d : test accuracy: %5.4f\n"%(itr, float(avg_test_acc)))'''

            feed_dict.update({self.model.is_training: True})
            self.sess.run([self.model.train_op], feed_dict=feed_dict)

            feed_dict.update({self.model.is_training: False})  # note: that's important
            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

            cur_iter = self.model.global_step_tensor.eval(self.sess)
            if cur_iter % self.config.TCov == 0:
                self.sess.run([self.model.cov_update_op], feed_dict=feed_dict)

            if cur_iter % self.config.TInv == 0:
                self.sess.run([self.model.inv_update_op, self.model.var_update_op], feed_dict=feed_dict)
        
        '''covs_lst = []
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if "u_c" in i.name or "v_c" in i.name :
                covs_lst.append(i)   # i.name if you want just a name'''

        print('---->saving ', cur_epoch+self.config.reload_step)
        checkpoint_path = os.path.join(self.checkpoint_dir, self.model_name)
        self.model.saver.save(self.sess, checkpoint_path, global_step=cur_epoch+self.config.reload_step)        

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("train | loss: %5.4f | accuracy: %5.4f"%(float(avg_loss), float(avg_acc)))

        # summarize
        summaries_dict = dict()
        summaries_dict['train_loss'] = avg_loss
        summaries_dict['train_acc'] = avg_acc

        # summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

        # self.model.save(self.sess)

    def test_epoch(self):
        loss_list = []
        acc_list = []
        for (x, y) in self.test_loader:
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: False,
                self.model.n_particles: self.config.test_particles
            }
            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("test | loss: %5.4f | accuracy: %5.4f\n"%(float(avg_loss), float(avg_acc)))

        # summarize
        summaries_dict = dict()
        summaries_dict['test_loss'] = avg_loss
        summaries_dict['test_acc'] = avg_acc

        # summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

    def check_grad(self):
        if self.config.reload_step >= 0 :
            print('---->reloading ', self.config.reload_step)
            self.reload(self.config.reload_step, self.sess, self.model.saver)
         
        (x, y) = next(iter(self.train_loader))

        num_samples = 600
        num_trials = 10

        opt = self.model.optim
        
        trainable_vars = tf.trainable_variables()
        gradient_step = opt.compute_gradients(self.model.total_loss, trainable_vars)

        feed_dict = {self.model.inputs: x, self.model.targets: y, self.model.is_training: True, 
                     self.model.n_particles: self.config.train_particles}
        
        if self.config.use_conv2d :
            W1_shape = [3, 3, 3, 32]
            W2_shape = [3, 3, 32, 32]
            W3_shape = [3, 3, 32, 64]
            W4_shape = [3, 3, 64, 64]
            W5_shape = [3, 3, 64, 128]
            W6_shape = [3, 3, 128, 128]
            W7_shape = [3, 3, 128, 128]
            W8_shape = [3, 3, 128, 256]
            W9_shape = [3, 3, 256, 256]
            W10_shape = [3, 3, 256, 256]
            W11_shape = [3, 3, 256, 256]
            W12_shape = [3, 3, 256, 256]
            W13_shape = [3, 3, 256, 256]
            W_FC_shape = [256, 10]

            W1_grad_var = np.zeros([num_trials])
            W2_grad_var = np.zeros([num_trials])
            W3_grad_var = np.zeros([num_trials])
            W4_grad_var = np.zeros([num_trials])
            W5_grad_var = np.zeros([num_trials])
            W6_grad_var = np.zeros([num_trials])
            W7_grad_var = np.zeros([num_trials])
            W8_grad_var = np.zeros([num_trials])
            W9_grad_var = np.zeros([num_trials])
            W10_grad_var = np.zeros([num_trials])
            W11_grad_var = np.zeros([num_trials])
            W12_grad_var = np.zeros([num_trials])
            W13_grad_var = np.zeros([num_trials])
            W_FC_grad_var = np.zeros([num_trials])

            for i in range(num_trials) :
                print('Iter {}/{}'.format(i, num_trials))
                W1_grad_lst = np.zeros([num_samples,W1_shape[0],W1_shape[1],W1_shape[2],W1_shape[3]])
                W2_grad_lst = np.zeros([num_samples,W2_shape[0],W2_shape[1],W2_shape[2],W2_shape[3]])
                W3_grad_lst = np.zeros([num_samples,W3_shape[0],W3_shape[1],W3_shape[2],W3_shape[3]])
                W4_grad_lst = np.zeros([num_samples,W4_shape[0],W4_shape[1],W4_shape[2],W4_shape[3]])
                W5_grad_lst = np.zeros([num_samples,W5_shape[0],W5_shape[1],W5_shape[2],W5_shape[3]])
                W6_grad_lst = np.zeros([num_samples,W6_shape[0],W6_shape[1],W6_shape[2],W6_shape[3]])
                W7_grad_lst = np.zeros([num_samples,W7_shape[0],W7_shape[1],W7_shape[2],W7_shape[3]])
                W8_grad_lst = np.zeros([num_samples,W8_shape[0],W8_shape[1],W8_shape[2],W8_shape[3]])
                W9_grad_lst = np.zeros([num_samples,W9_shape[0],W9_shape[1],W9_shape[2],W9_shape[3]])
                W10_grad_lst = np.zeros([num_samples,W10_shape[0],W10_shape[1],W10_shape[2],W10_shape[3]])
                W11_grad_lst = np.zeros([num_samples,W11_shape[0],W11_shape[1],W11_shape[2],W11_shape[3]])
                W12_grad_lst = np.zeros([num_samples,W12_shape[0],W12_shape[1],W12_shape[2],W12_shape[3]])
                W13_grad_lst = np.zeros([num_samples,W13_shape[0],W13_shape[1],W13_shape[2],W13_shape[3]])
                W_FC_grad_lst = np.zeros([num_samples,W_FC_shape[0],W_FC_shape[1]])

                for j in range(num_samples) :
                    grad_W = self.sess.run(gradient_step, feed_dict=feed_dict)
                    W1_grad_lst[j,:,:,:,:] = grad_W[0][0]
                    W2_grad_lst[j,:,:,:,:] = grad_W[2][0]
                    W3_grad_lst[j,:,:,:,:] = grad_W[4][0]
                    W4_grad_lst[j,:,:,:,:] = grad_W[6][0]
                    W5_grad_lst[j,:,:,:,:] = grad_W[8][0]
                    W6_grad_lst[j,:,:,:,:] = grad_W[10][0]
                    W7_grad_lst[j,:,:,:,:] = grad_W[12][0]
                    W8_grad_lst[j,:,:,:,:] = grad_W[14][0]
                    W9_grad_lst[j,:,:,:,:] = grad_W[16][0]
                    W10_grad_lst[j,:,:,:,:] = grad_W[18][0]
                    W11_grad_lst[j,:,:,:,:] = grad_W[20][0]
                    W12_grad_lst[j,:,:,:,:] = grad_W[22][0]
                    W13_grad_lst[j,:,:,:,:] = grad_W[24][0]
                    W_FC_grad_lst[j,:,:] = grad_W[26][0]

                W1_grad_var[i] = np.mean(np.var(W1_grad_lst, axis=0))
                W2_grad_var[i] = np.mean(np.var(W2_grad_lst, axis=0))
                W3_grad_var[i] = np.mean(np.var(W3_grad_lst, axis=0))
                W4_grad_var[i] = np.mean(np.var(W4_grad_lst, axis=0))
                W5_grad_var[i] = np.mean(np.var(W5_grad_lst, axis=0))
                W6_grad_var[i] = np.mean(np.var(W6_grad_lst, axis=0))
                W7_grad_var[i] = np.mean(np.var(W7_grad_lst, axis=0))
                W8_grad_var[i] = np.mean(np.var(W8_grad_lst, axis=0))
                W9_grad_var[i] = np.mean(np.var(W9_grad_lst, axis=0))
                W10_grad_var[i] = np.mean(np.var(W10_grad_lst, axis=0))
                W11_grad_var[i] = np.mean(np.var(W11_grad_lst, axis=0))
                W12_grad_var[i] = np.mean(np.var(W12_grad_lst, axis=0))
                W13_grad_var[i] = np.mean(np.var(W13_grad_lst, axis=0))
                W_FC_grad_var[i] = np.mean(np.var(W_FC_grad_lst, axis=0))

            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W1 gradients has variance: \n",W1_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W2 gradients has variance: \n",W2_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W3 gradients has variance: \n",W3_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W4 gradients has variance: \n",W4_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W5 gradients has variance: \n",W5_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W6 gradients has variance: \n",W6_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W7 gradients has variance: \n",W7_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W8 gradients has variance: \n",W8_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W9 gradients has variance: \n",W9_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W10 gradients has variance: \n",W10_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W11 gradients has variance: \n",W11_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W12 gradients has variance: \n",W12_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W13 gradients has variance: \n",W13_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W_FC gradients has variance: \n",W_FC_grad_var)

            grad_save_path = '{}/batch{}'.format(GRAD_CHECK_ROOT_DIR, self.config.batch_size)
            if not os.path.exists(grad_save_path):
                os.makedirs(grad_save_path)

            if self.config.use_flip :
                with open('{}/ptb_var_62_train_acc_conv_flip.pkl'.format(grad_save_path), 'wb') as f2:
                    pickle.dump([W1_grad_var, W2_grad_var, W3_grad_var, W4_grad_var, W5_grad_var, W6_grad_var, W7_grad_var, W8_grad_var, W9_grad_var, W10_grad_var, W11_grad_var, W12_grad_var, W13_grad_var, W_FC_grad_var], f2)
                    print('======================save_flip_model_batch_size_{}========================='.format(self.config.batch_size))
            else :
                with open('{}/ptb_var_62_train_acc_conv_pert.pkl'.format(grad_save_path), 'wb') as f1:
                    pickle.dump([W1_grad_var, W2_grad_var, W3_grad_var, W4_grad_var, W5_grad_var, W6_grad_var, W7_grad_var, W8_grad_var, W9_grad_var, W10_grad_var, W11_grad_var, W12_grad_var, W13_grad_var, W_FC_grad_var], f1)
                    print('======================save_pert_model_batch_size_{}========================='.format(self.config.batch_size))
        else :
            W1_shape = [784, 512]
            W2_shape = [512, 512]
            W3_shape = [512, 512]
            W4_shape = [512, 512]
            W5_shape = [512, 10]

            W1_grad_var = np.zeros([num_trials])
            W2_grad_var = np.zeros([num_trials])
            W3_grad_var = np.zeros([num_trials])
            W4_grad_var = np.zeros([num_trials])
            W5_grad_var = np.zeros([num_trials])


            for i in range(num_trials) :
                print('Iter {}/{}'.format(i, num_trials))
                W1_grad_lst = np.zeros([num_samples,W1_shape[0],W1_shape[1]])
                W2_grad_lst = np.zeros([num_samples,W2_shape[0],W2_shape[1]])
                W3_grad_lst = np.zeros([num_samples,W3_shape[0],W3_shape[1]])
                W4_grad_lst = np.zeros([num_samples,W4_shape[0],W4_shape[1]])
                W5_grad_lst = np.zeros([num_samples,W5_shape[0],W5_shape[1]])

                for j in range(num_samples) :
                    grad_W = self.sess.run(gradient_step, feed_dict=feed_dict)
                    W1_grad_lst[j,:,:] = grad_W[0][0]
                    W2_grad_lst[j,:,:] = grad_W[2][0]
                    W3_grad_lst[j,:,:] = grad_W[4][0]
                    W4_grad_lst[j,:,:] = grad_W[6][0]
                    W5_grad_lst[j,:,:] = grad_W[8][0]

                W1_grad_var[i] = np.mean(np.var(W1_grad_lst, axis=0))
                W2_grad_var[i] = np.mean(np.var(W2_grad_lst, axis=0))
                W3_grad_var[i] = np.mean(np.var(W3_grad_lst, axis=0))
                W4_grad_var[i] = np.mean(np.var(W4_grad_lst, axis=0))
                W5_grad_var[i] = np.mean(np.var(W5_grad_lst, axis=0))

            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W1 gradients has variance: \n",W1_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W2 gradients has variance: \n",W2_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W3 gradients has variance: \n",W3_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W4 gradients has variance: \n",W4_grad_var)
            print("Batch size: ",str(self.config.batch_size)," With flip: ",str(self.config.use_flip),", W5 gradients has variance: \n",W5_grad_var)


            grad_save_path = '{}/batch{}'.format(GRAD_CHECK_ROOT_DIR, self.config.batch_size)
            if not os.path.exists(grad_save_path):
                os.makedirs(grad_save_path)

            if self.config.use_flip :
                with open('{}/ptb_var_87_train_acc_flip.pkl'.format(grad_save_path), 'wb') as f2:
                    pickle.dump([W1_grad_var, W2_grad_var, W3_grad_var, W4_grad_var], f2)
                    print('======================save_flip_model_batch_size_{}========================='.format(self.config.batch_size))
            else :
                with open('{}/ptb_var_87_train_acc_pert.pkl'.format(grad_save_path), 'wb') as f1:
                    pickle.dump([W1_grad_var, W2_grad_var, W3_grad_var, W4_grad_var], f1)
                    print('======================save_pert_model_batch_size_{}========================='.format(self.config.batch_size))


    def reload(self, step, sess, saver) :
        checkpoint_path = os.path.join(self.checkpoint_dir, self.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta') :
            print('------- no such checkpoint', model_path)
            return
        print('---->restoring ', step)
        saver.restore(sess, model_path)