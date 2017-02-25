#!/usr/bin/env python
"""Train a model to predict actions given observations
   Parameters: Filename of pickled dictionary with keys "observations", "actions"
"""


import pickle
import tensorflow as tf
import numpy as np


def model(observations, actions, dirName):
    Label_Dimension=actions.shape[1]
    Feature_Dimension=observations.shape[1]

    feature_column = [tf.contrib.layers.real_valued_column("", dimension=Feature_Dimension)]

    estimator = tf.contrib.learn.DNNRegressor(
                      feature_columns=feature_column,
                      hidden_units=[64,64],
                      #activation_fn = tf.nn.relu,
                      activation_fn = tf.tanh,
                      model_dir = "./trained/"+dirName,
                      label_dimension=Label_Dimension
                )

    def input_fn_train(): # returns x, Y       
        return {"": tf.constant(observations, dtype=tf.float32)}, \
               tf.constant(actions, dtype=tf.float32)

    print("shape of observations", observations.shape, "shape of actions ", actions.shape)
    estimator.fit(input_fn=input_fn_train, steps = 3000)
    print("fitting done")
    # check how well the model fits the training data
    ev = estimator.evaluate(input_fn=input_fn_train, steps = 2)
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

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

    tf.logging.set_verbosity(tf.logging.INFO)

    model(observations, actions, args.expert_policy_file)

if __name__ == '__main__':
    main()
