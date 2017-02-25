import tensorflow as tf
import numpy as np

def play(estimator, env, num_episodes, render = True):
    returns = []
    observations = []
    actions = []
    for i in range(num_episodes):
      obs = env.reset()
      done = False
      totalr = 0
      steps = 0
      while not done:
          action = estimator.predict(obs[None,:], as_iterable=False)
          print(" obs ", obs[None,:]," :: action ", action)
          observations.append(obs)
          actions.append(action)
          obs, r, done, _ = env.step(action)
          totalr += r
          steps += 1
          if steps % 10 == 0:
              if render:
                 env.render()
      returns.append(totalr)
    return returns


def loadEstimator(dirName, Label_Dimension, Feature_Dimension):
    feature_column = [tf.contrib.layers.real_valued_column("", dimension=Feature_Dimension)]
    estimator = tf.contrib.learn.DNNRegressor(
                      feature_columns=feature_column,
                      hidden_units=[64,64],
                      #activation_fn = tf.nn.relu,
                      activation_fn = tf.tanh,
                      model_dir = dirName,
                      label_dimension=Label_Dimension
                )
    return estimator

def loadDimensions(envName):
    import pickle
    data = pickle.load(open("./trainingData/"+envName+".p", "rb"))

    observations = data["observations"]
    actions = np.squeeze(data["actions"])
    Label_Dimension=actions.shape[1]
    Feature_Dimension=observations.shape[1]
    return Label_Dimension, Feature_Dimension


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envName', type=str)
    parser.add_argument('numEpisodes', type=int)

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    dirName = "./trained/"+args.envName+".p"
    print("Loading trianed model from",dirName)

    label_dimension, feature_dimension = loadDimensions(args.envName)
    estimator = loadEstimator(dirName, label_dimension, feature_dimension)

    import gym
    env = gym.make(args.envName)
    play(estimator, env, args.numEpisodes, True)

if __name__ == '__main__':
    main()
