import tensorflow as tf
import pandas as pd
import numpy as np
from dataloader import DataLoader
from tensorflow.python.platform import gfile
from time import strftime, localtime, time


### properties
# General
# TODO : declare additional properties
# not fixed (change or add property as you like)
batch_size = 32
epoch_num = 100


# fixed
metadata_path = 'dataset/track_metadata.csv'
# True if you want to train, False if you already trained your model
### TODO : IMPORTANT !!! Please change it to False when you submit your code
is_train_mode = False
### TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
checkpoint_path = 'checkpoint'
# 'track_genre_top' for project 1, 'listens' for project 2
label_column_name = 'track_genre_top'


# Placeholder and variables
# TODO : declare placeholder and variables

# Build model
# TODO : build your model here

# Loss and optimizer
# TODO : declare loss and optimizer operation
# Train and evaluate
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()


    if is_train_mode:
        train_dataloader = DataLoader(file_path='dataset/track_metadata.csv', batch_size=batch_size,
                                      label_column_name=label_column_name, is_training=True)

        for epoch in range(epoch_num):
            total_batch = train_dataloader.num_batch

            for i in range(total_batch):
                batch_x, batch_y = train_dataloader.next_batch()
                # TODO:  do some train step code here

        print('Training finished !')
        output_dir = checkpoint_path + '/run-%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5]
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        saver.save(sess, output_dir)
        print('Model saved in file : %s'%output_dir)
    else:
        # skip training and restore graph for validation test
        saver.restore(sess, checkpoint_path)


    # Validation
    validation_dataloader = DataLoader(file_path='dataset/track_metadata.csv', batch_size = batch_size, label_column_name = label_column_name, is_training= False)

    average_val_cost = 0
    for i in range(validation_dataloader.num_batch):
        batch_x, batch_y = validation_dataloader.next_batch()
        # TODO : do some loss calculation here
        # average_cost += loss/validation_dataloader.num_batch

    print('Validation loss : %f'%average_val_cost)

    # accuracy test example
    # TODO :
    # pred = tf.nn.softmax(<your network output logit object>)
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # avg_accuracy = 0
    #for i in range(validation_dataloader.num_batch):
        #batch_x, batch_y = validation_dataloader.next_batch()
        #acc = accuracy_op.eval({x:batch_x, y: batch_y})
        # avg_accuracy += acc / validation_dataloader.num_batch
    # print("Average accuracy on validation set ", avg_accuracy)




















