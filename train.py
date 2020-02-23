import argparse
import os
import time

import numpy as np
import tensorflow as tf

from model import Model
from dataset import OCR_DataLoader, map_and_count

from config import args

with open(args.table_path, "r") as f:
    INT_TO_CHAR = [char.strip() for char in f]
NUM_CLASSES = len(INT_TO_CHAR)
BLANK_INDEX = NUM_CLASSES - 1  # Make sure the blank index is what.


def create_dataloader():
    train_dl = OCR_DataLoader(
        annotation_paths=args.train_annotation_paths,
        image_height=args.image_height,
        image_width=args.image_width,
        table_path=args.table_path,
        blank_index=BLANK_INDEX,
        shuffle=True,
        batch_size=args.batch_size,
        rgb=args.rgb,
        keep_ratio_with_pad=args.keep_ratio_with_pad,
    )
    print(f"Num of training samples: {len(train_dl)}")
    if args.val_annotation_paths:
        val_dl = OCR_DataLoader(
            annotation_paths=args.val_annotation_paths,
            image_height=args.image_height,
            image_width=args.image_width,
            table_path=args.table_path,
            blank_index=BLANK_INDEX,
            batch_size=args.batch_size,
            rgb=args.rgb,
            keep_ratio_with_pad=args.keep_ratio_with_pad,
        )
        print(f"Num of val samples: {len(val_dl)}")
    else:
        val_dl = None
        print("No validation dataset")

    # TODO
    print(f"Num of classes: {NUM_CLASSES}")
    print(f"Blank index is {BLANK_INDEX}")
    return train_dl, val_dl


@tf.function
def val_one_step(model, x, y):
    logits = model(x, training=False)
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    loss = tf.nn.ctc_loss(
        labels=y,
        logits=logits,
        label_length=None,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=BLANK_INDEX)
    loss = tf.reduce_mean(loss)
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length,
        merge_repeated=True)
    return loss, decoded


def val(model, dataset, step, num_samples):
    avg_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
    num_correct_samples = 0
    for x, y in dataset:
        loss, decoded = val_one_step(model, x, y)
        cnt = map_and_count(decoded, y, INT_TO_CHAR)
        avg_loss.update_state(loss)
        num_correct_samples += cnt
    tf.summary.scalar("loss", avg_loss.result(), step=step)
    accuracy = num_correct_samples / num_samples * 100
    tf.summary.scalar("accuracy", accuracy, step=step)
    avg_loss.reset_states()


@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
        loss = tf.nn.ctc_loss(
            labels=y,
            logits=logits,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=False,
            # TODO
            blank_index=BLANK_INDEX)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(model, optimizer, dataset, log_freq=10):
    avg_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
    for x, y in dataset:
        loss = train_one_step(model, optimizer, x, y)
        avg_loss.update_state(loss)
        if tf.equal(optimizer.iterations % log_freq, 0):
            tf.summary.scalar("loss", avg_loss.result(),
                              step=optimizer.iterations)
            avg_loss.reset_states()


def workflow():
    """ dataset preparation """
    # TODO
    train_dl, val_dl = create_dataloader()

    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(f"Start at {localtime}")

    """ model configuration """
    # TODO
    # if 'CTC' in args.Prediction:
    #     converter = CTCLabelConverter(args.character)
    # else:
    #     converter = AttnLabelConverter(args.character)

    # args.num_class = len(converter.character)
    args.num_class = NUM_CLASSES

    model = Model(args)
    # model.summary()
    # weight initialization
    # data parallel for multi-GPU

    """ 
    setup loss
    在train_one_step函数中设置，目前只考虑CTC Loss 
    """
    # setup optimizer
    # TODO: lr_schedule 报错
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)
    if args.adam:
        # TODO: Adam优化器 W tensorflow/core/util/ctc/ctc_loss_calculator.cc:145] No valid path found.
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=args.learning_rate, rho=args.rho, epsilon=args.epsilon)
    print("Optimizer:")
    print(optimizer)

    if not args.experiment_name:
        args.experiment_name = f'{args.Transformation}-{args.FeatureExtraction}-{args.SequenceModeling}-{args.Prediction}'
    os.makedirs(f'./saved_models/{args.experiment_name}', exist_ok=True)
    """ final options """
    # print(args)

    """ start training """
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # if args.checkpoint:
    #     localtime = args.checkpoint.rstrip("/").split("/")[-1]
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=f'./saved_models/{args.experiment_name}',
        max_to_keep=args.max_to_keep)
    checkpoint.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch")

    train_summary_writer = tf.summary.create_file_writer(
        f"logs/{args.experiment_name}/train")
    val_summary_writer = tf.summary.create_file_writer(
        f"logs/{args.experiment_name}/val")

    for epoch in range(1, args.epochs + 1):
        with train_summary_writer.as_default():
            train(model, optimizer, train_dl())
        if not (epoch - 1) % args.save_freq:
            checkpoint_path = manager.save(optimizer.iterations)
            print(f"Model saved to {checkpoint_path}")
            if val_dl is not None:
                with val_summary_writer.as_default():
                    val(model, val_dl(), optimizer.iterations, len(val_dl))


if __name__ == '__main__':
    workflow()
