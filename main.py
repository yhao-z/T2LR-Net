# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import argparse
import scipy.io as scio
import numpy as np
import time

from tools import mse, calc_PSNR, calc_SNR
from mask_generator import generate_mask
from model import TLR_Net
from dataset_tfrecord import singCoil_parse_function


def get_dataset_singCoil(mode, batch_size, shuffle=False):
    filenames = './data/OCMR_3T_train.tfrecord'

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(singCoil_parse_function)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=50)
    dataset = dataset.batch(batch_size)

    return dataset


def get_testdata():
    filenames = './data/OCMR_singCoil_test.tfrecord'

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(singCoil_parse_function)
    dataset = dataset.batch(1)

    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['50'], help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--learning_rate', metavar='float', nargs=1, default=['0.001'], help='initial learning rate')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['15'], help='number of network iterations')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'], help='GPU No.')
    parser.add_argument('--weight', metavar='str', nargs=1, default=None)

    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    mode = 'train'
    masktype = 'radial_16'
    
    niter = int(args.niter[0])
    batch_size = int(args.batch_size[0])
    num_epoch = int(args.num_epoch[0])
    learning_rate = float(args.learning_rate[0])
    weight_file = args.weight

    modeldir = './models/'
    os.makedirs(modeldir) if not os.path.exists('./models/') else None

    # prepare dataset
    dataset = get_dataset_singCoil(mode, batch_size, shuffle=True)
    tf.print('dataset loaded.')

    # initialize network
    net = TLR_Net(niter)
    if weight_file is not None:
        tf.print('load weights.')
        net.load_weights(weight_file)
    tf.print('network initialized.')

    learning_rate_org = learning_rate
    learning_rate_decay = 0.95

    optimizer = tf.optimizers.Adam(learning_rate_org)

    # Iterate over epochs.
    total_step = 0
    param_num = 0
    loss = 0

    for epoch in range(num_epoch):
        for step, sample in enumerate(dataset):
            # forward
            t0 = time.time()
            k0 = None
            with tf.GradientTape() as tape:
                k0, label = sample

                if k0 is None:
                    continue
                if k0.shape[0] < batch_size:
                    continue

                nb, nt, nx, ny = k0.get_shape()
                mask = generate_mask([nx, ny, nt], float(masktype.split('_', 1)[1]), masktype.split('_', 1)[0])
                mask = np.transpose(mask, (2, 0, 1))
                mask = tf.constant(np.complex64(mask + 0j))

                k0 = k0 * mask
                recon = net(k0, mask)
                recon_abs = tf.abs(recon)

                loss = mse(recon, label)

            # backward
            grads = tape.gradient(loss, net.trainable_weights)  ####################################
            optimizer.apply_gradients(zip(grads, net.trainable_weights))  #################################

            # calculate parameter number
            if total_step == 0:
                param_num = np.sum([np.prod(v.get_shape()) for v in net.trainable_variables])
                print('parameters: ', param_num)

            # log output
            if step % 100 == 0:
                tf.print('Epoch', epoch + 1, '/', num_epoch, 'Step', step, 'loss =', loss.numpy(), 'time',
                        time.time() - t0, 'lr = ', learning_rate)
            total_step += 1

        # learning rate decay for each epoch
        learning_rate = learning_rate_org * learning_rate_decay ** (epoch + 1)  # (total_step / decay_steps)
        optimizer = tf.optimizers.Adam(learning_rate)

        # save model each epoch
        if epoch in [0, num_epoch-1, num_epoch]:
            model_epoch_dir = os.path.join(modeldir, 'epoch-' + str(epoch + 1), 'ckpt')
            net.save_weights(model_epoch_dir, save_format='tf')
    ########################################################################
    # test and print results
    print('######################   test...    ######################')
    testdata = get_testdata()
    tf.print('testdata loaded.')
    SNRs = []
    PSNRs = []
    SSIMs = []
    MSEs = []
    for step, sample in enumerate(testdata):
        k0, label = sample
        maskdir = './data/test_mask_radial_16/'
        mask = scio.loadmat(maskdir+'/mask%d.mat' % (step+1))['sampling_mask']
        mask = np.transpose(mask, (2, 0, 1))
        mask = tf.constant(np.complex64(mask + 0j))
        k0 = k0 * mask

        t0 = time.time()
        recon = net(k0, mask)
        t_end = time.time()
        SNR_ = calc_SNR(recon, label)
        PSNR_ = calc_PSNR(recon, label)
        SSIM_ = tf.image.ssim(tf.transpose(tf.abs(recon), [0, 2, 3, 1]), tf.transpose(tf.abs(label), [0, 2, 3, 1]), max_val=1.0)
        MSE_ = mse(recon, label)
        SNRs.append(SNR_)
        PSNRs.append(PSNR_)
        SSIMs.append(SSIM_)
        MSEs.append(MSE_)
        print('data %d --> SNR = \%.3f\, PSNR = \%.3f\, SSIM = \%.3f\, MSE = {%.3e}, t = %.1f' % (step, SNR_, PSNR_, SSIM_, MSE_, t_end-t0))
    PSNRs = np.array(PSNRs)
    MSEs = np.array(MSEs)
    print('SNR = %.3f(%.3f), PSNR = %.3f(%.3f), SSIM = %.3f(%.3f), MSE = %.3e(%.3e)' % (np.mean(SNRs), np.std(SNRs), np.mean(PSNRs), np.std(PSNRs), np.mean(SSIMs), np.std(SSIMs), np.mean(MSEs), np.std(MSEs)))


    
    
