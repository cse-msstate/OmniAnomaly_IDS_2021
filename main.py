# -*- coding: utf-8 -*-
import logging
import os
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
from pprint import pformat, pprint
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
import tensorflow as tf

from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict, register_config_arguments, Config
from tfsnippet.utils import set_random_seed

from omni_anomaly.eval_methods import pot_eval, bf_search
from omni_anomaly.model import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.training import Trainer
from omni_anomaly.utils import get_data_dim, get_data, save_z, preprocess

### Code Added ###
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import auc,roc_curve, precision_recall_curve, average_precision_score

#set_random_seed(123)
### Code Added ###

### Code Added ###
def dataLoading(path):
    # loading data
    df = pd.read_csv(path,dtype=np.float32)
    labels = df['label_class']

    x_train_df = df.drop(['label_class'], axis=1)
    x_train = x_train_df.values
    print(x_train.shape)

    return x_train, labels



class ExpConfig(Config):
    # dataset configuration

    ### Code Added ###
    filename = 'nslkdd_100'
    ### Code Added ###

    x_dim = None

    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = 3
    rnn_cell = 'GRU'  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = 'nf'  # 'nf' or None
    nf_layers = 20  # for nf
    max_epoch = 100000000000
    train_start = 0
    max_train_size = None  # `None` means full train set
    batch_size = 50
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None  # `None` means full test set

    # the range and step-size for score for searching best-f1
    # may vary for different dataset
    bf_search_min = -1500.
    bf_search_max = 1500.
    bf_search_step_size = 1.

    valid_step_freq = 100
    gradient_clip_norm = 10.

    early_stop = False  # whether to apply early stop method

    # pot parameters
    # recommend values for `level`:
    # SMAP: 0.07
    # MSL: 0.01
    # SMD group 1: 0.0050
    # SMD group 2: 0.0075
    # SMD group 3: 0.0001
    level = 0.01

    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = False  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    save_dir = filename
    restore_dir = None  # If not None, restore variables from this dir
    result_dir = filename  # Where to save the result file
    train_score_filename = filename+'_train_score.pkl'
    test_score_filename = filename+' test_score.pkl'


def main():
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )



        # prepare the data
        #(x_train, _), (x_test, y_test) = \
            #get_data(config.dataset, config.max_train_size, config.max_test_size, train_start=config.train_start,
                     #test_start=config.test_start)

    # construct the model under `variable_scope` named 'model'
    model_name = 'model' + str(acc)
    with tf.variable_scope(model_name) as model_vs:
        model = OmniAnomaly(config=config, name="model")

        # construct the trainer
        trainer = Trainer(model=model,
                          model_vs=model_vs,
                          max_epoch=config.max_epoch,
                          batch_size=config.batch_size,
                          valid_batch_size=config.test_batch_size,
                          initial_lr=config.initial_lr,
                          lr_anneal_epochs=config.lr_anneal_epoch_freq,
                          lr_anneal_factor=config.lr_anneal_factor,
                          grad_clip_norm=config.gradient_clip_norm,
                          valid_step_freq=500,#config.valid_step_freq,
                          x_test=x_test,
                          y_test=y_test)


        # construct the predictor
        predictor = Predictor(model, batch_size=config.batch_size, n_z=config.test_n_z,
                              last_point_only=True)

        with tf.Session().as_default():

            if config.restore_dir is not None:
                # Restore variables from `save_dir`.
                saver = VariableSaver(get_variables_as_dict(model_vs), config.restore_dir)
                saver.restore()

            if config.max_epoch > 0:
                # train the model
                train_start = time.time()
                best_valid_metrics = trainer.fit(x_train)
                train_time = (time.time() - train_start) / config.max_epoch
                best_valid_metrics.update({
                    'train_time': train_time
                })
            else:
                best_valid_metrics = {}

            # get score of train set for POT algorithm
            train_score, train_z, train_pred_speed = predictor.get_score(x_train)
            train_roc_auc = aucPerformance(train_score, y_train[99:], 'train')
            train_pr_auc = prcPerformance(train_score, y_train[99:])

            train_score.tofile('train_scores.csv', sep=',')

            if config.train_score_filename is not None:
                with open(os.path.join(config.result_dir, config.train_score_filename), 'wb') as file:
                    pickle.dump(train_score, file)
            if config.save_z:
                save_z(train_z, 'train_z')

            if x_test is not None:
                # get score of test set
                test_start = time.time()
                test_score, test_z, pred_speed = predictor.get_score(x_test)
                test_time = time.time() - test_start

                ### CODE ADDED ###
                roc_auc = aucPerformance(test_score, y_test[99:], 'test')
                pr_auc = prcPerformance(test_score, y_test[99:])
                ### CODE ADDED ###

                if config.save_z:
                    save_z(test_z, 'test_z')
                best_valid_metrics.update({
                    'pred_time': pred_speed,
                    'pred_total_time': test_time
                })
                if config.test_score_filename is not None:
                    with open(os.path.join(config.result_dir, config.test_score_filename), 'wb') as file:
                        pickle.dump(test_score, file)

                if y_test is not None and len(y_test) >= len(test_score):
                    if config.get_score_on_dim:
                        # get the joint score
                        test_score = np.sum(test_score, axis=-1)
                        train_score = np.sum(train_score, axis=-1)

                    # get best f1
                    t, th = bf_search(test_score, y_test[-len(test_score):],
                                      start=config.bf_search_min,
                                      end=config.bf_search_max,
                                      step_num=int(abs(config.bf_search_max - config.bf_search_min) /
                                                   config.bf_search_step_size),
                                      display_freq=100)
                    # get pot results
                    pot_result = pot_eval(train_score, test_score, y_test[-len(test_score):], level=config.level)

                    # output the results
                    best_valid_metrics.update({
                        'best-f1': t[0],
                        'precision': t[1],
                        'recall': t[2],
                        'TP': t[3],
                        'TN': t[4],
                        'FP': t[5],
                        'FN': t[6],
                        'latency': t[-1],
                        'threshold': th,
                        'roc_auc': roc_auc,
                        'pr_auc': pr_auc
                    })
                    best_valid_metrics.update(pot_result)
                results.update_metrics(best_valid_metrics)

            if config.save_dir is not None:
                # save the variables
                var_dict = get_variables_as_dict(model_vs)
                saver = VariableSaver(var_dict, config.save_dir)
                saver.save()
            print('=' * 30 + 'result' + '=' * 30)
            pprint(best_valid_metrics)


def aucPerformance(mse, labels, name):
    fpr, tpr, thresholds = roc_curve(labels, mse, pos_label = 1 )
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0,1],[0,1],'r--')
    # plt.xlim([-0.001, 1])
    # plt.ylim([0, 1.001])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig(str(name)+'.png')

    return roc_auc

def prcPerformance(scores, labels, show_graph=True):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    print(pr_auc)

    return pr_auc


if __name__ == '__main__':

    filename = str(sys.argv[1])
    X, y = dataLoading("./data/" + filename + ".csv")

    ### CODE ADDED FOR SURVEY ###
    X = preprocess(X)
    acc = 0

    # names of folds that change the name of the results file
    test_names = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']

    tscv = TimeSeriesSplit(n_splits=4)
    for train_index, test_index in tscv.split(X):
        # check that we are on the correct fold
        if acc == int(sys.argv[2]):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print(x_train.shape, x_test.shape)
            print(y_train.shape, y_test.shape)

            config = ExpConfig()
            config.filename = test_names[acc]
            config.x_dim = x_train.shape[1]
            config.save_dir = test_names[acc]
            config.result_dir = test_names[acc]
            # get config obj


            print_with_title('Configurations', pformat(config.to_dict()), after='\n')

            # open the result object and prepare for result directories if specified
            results = MLResults(config.result_dir)
            #results.save_config(config)  # save experiment settings for review
            results.make_dirs(config.save_dir, exist_ok=True)
            with warnings.catch_warnings():
                # suppress DeprecationWarning from NumPy caused by codes in TensorFlow-Probability
                warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')
                main_start_time = time.time()
                main()
                print("END OF", test_names[acc], ": ", (time.time() - main_start_time))
                acc += 1
            break
        else:
            print("skip fold")
            acc += 1
