#!/usr/bin/env python
#coding=utf-8
import pdb
import random

import paddle.v2 as paddle
from paddle.v2.layer import parse_network


def define_slice_layer():
    input_seq = paddle.layer.data(
        name="sequence", type=paddle.data_type.dense_vector_sequence(3))
    starts = paddle.layer.data(
        name="start_ids", type=paddle.data_type.dense_vector(1))
    ends = paddle.layer.data(
        name="end_ids", type=paddle.data_type.dense_vector(1))
    return paddle.layer.seq_slice(input=input_seq, starts=starts, ends=ends)


def gen_test_data():
    BATCH_SIZE = 5
    MIN_SEQ_NUM = 5
    MAX_SEQ_NUM = 15

    data_batch = []
    for i in range(BATCH_SIZE):
        seq_len = random.randint(MIN_SEQ_NUM, MAX_SEQ_NUM)

        selected_ids = float(random.randint(0, seq_len - 1))
        seq_input = [[float(i)] * 3 for i in range(seq_len)]

        data_batch.append([seq_input, [selected_ids], [selected_ids]])
    return data_batch


def test_slice_layer():
    paddle.init(use_gpu=True, trainer_count=1)
    sliced = define_slice_layer()

    parameters = paddle.parameters.create(sliced)

    inferer = paddle.inference.Inference(
        output_layer=sliced, parameters=parameters)

    test_batch = gen_test_data()
    outs = inferer.infer(
        input=test_batch, flatten_result=False, field="value")[0]

    for i, test_sample in enumerate(test_batch):
        print("\nid = %d" % int(test_sample[-1][0]))
        print outs[i, :]


if __name__ == "__main__":
    # print parse_network(define_slice_layer())
    test_slice_layer()
