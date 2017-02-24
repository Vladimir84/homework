#!/usr/bin/env python

import pickle
import tensorflow as tf
import numpy as np


def model(observations, actions):
    feature_column = [tf.contrib.layers.real_valued_column("", dimension=111)]

    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = tf.contrib.learn.DNNRegressor(
           feature_columns=feature_column,
           hidden_units=[111,64,8],
           activation_fn = tf.nn.relu,
      #     model_dir = "/tmp/Ant-v1",
      #     enable_centered_bias=False,
           label_dimension=8
      #     dropout = 0.5
           )

    #estimator.fit(observations, actions, steps=2000)

    def input_fn_train(): # returns x, Y       
        return {"": tf.constant(
            observations, dtype=tf.float32)
               }, tf.constant(
                       actions, dtype=tf.float32)

    print("shape of observations", observations.shape, "shape of actions ", actions.shape)
    estimator.fit(input_fn=input_fn_train, steps = 1000)
    print("fitting done")
    ev = estimator.evaluate(input_fn=input_fn_train, steps = 2)
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))
    print("Done")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)

    args = parser.parse_args()

    data = pickle.load(open(args.expert_policy_file, "rb"))

    observations = data["observations"]
    actions = np.squeeze(data["actions"])
    print("observations.shape ", observations.shape)
    print("actions.shape ", actions.shape)

    model(observations, actions)

if __name__ == '__main__':
    main()
