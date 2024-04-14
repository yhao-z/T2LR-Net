# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import argparse
import scipy.io as scio
import numpy as np
import time

from tools import mse, calc_PSNR, calc_SNR
from model import TLR_Net
from dataset_tfrecord import singCoil_parse_function


def get_testdata():
    filenames = './data/OCMR_singCoil_test.tfrecord'

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(singCoil_parse_function)
    dataset = dataset.batch(1)

    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', metavar='int', nargs=1, default=['15'], help='number of network iterations')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'], help='GPU No.')
    parser.add_argument('--weight', metavar='str', nargs=1, default='./models/ckpt')

    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    masktype = 'radial_16'
    
    niter = int(args.niter[0])
    weight_file = args.weight

    # initialize network
    net = TLR_Net(niter)
    tf.print('load weights.')
    net.load_weights(weight_file)
    tf.print('network initialized.')

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

    
    
