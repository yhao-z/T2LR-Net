# -*- coding: utf-8 -*-
import tensorflow as tf


def singCoil_parse_function(example_proto):
    dics = {'k_real': tf.io.VarLenFeature(dtype=tf.float32),
            'k_imag': tf.io.VarLenFeature(dtype=tf.float32),
            'label_real': tf.io.VarLenFeature(dtype=tf.float32),
            'label_imag': tf.io.VarLenFeature(dtype=tf.float32),
            'k_shape': tf.io.VarLenFeature(dtype=tf.int64),
            'label_shape': tf.io.VarLenFeature(dtype=tf.int64)}
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    parsed_example['k_real'] = tf.sparse.to_dense(parsed_example['k_real'])
    parsed_example['k_imag'] = tf.sparse.to_dense(parsed_example['k_imag'])
    parsed_example['label_real'] = tf.sparse.to_dense(parsed_example['label_real'])
    parsed_example['label_imag'] = tf.sparse.to_dense(parsed_example['label_imag'])
    parsed_example['k_shape'] = tf.sparse.to_dense(parsed_example['k_shape'])
    parsed_example['label_shape'] = tf.sparse.to_dense(parsed_example['label_shape'])

    k = tf.complex(parsed_example['k_real'], parsed_example['k_imag'])
    label = tf.complex(parsed_example['label_real'], parsed_example['label_imag'])

    k = tf.reshape(k, parsed_example['k_shape'])
    label = tf.reshape(label, parsed_example['label_shape'])

    return k, label
