import tensorflow as tf
import numpy as np



def runEpisode(policy_fn, env, render = True):
    observations = []
    actions = []
    done = False
    steps = 0

    obs = env.reset()
    while not done:
        action = policy_fn(obs[None,:])
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        steps += 1
        if steps % 10 == 0:
            if render:
               env.render()
    return observations, actions

def trainedPolicy(estimator):
    def prediction(obs):
        return estimator.predict(obs, as_iterable=False)
    return prediction

def askExpert(expertPolicy, observations):
    expertActions = []
    for observation in observations:
        action = expertPolicy(observation)
        expertActions.append(action)
    return expertActions

def trainModel(estimator, observations, actions):
    def input_fn_train(): # returns x, Y       
        return {"": tf.constant(observations, dtype=tf.float32)}, \
               tf.constant(actions, dtype=tf.float32)

    estimator.fit(input_fn=input_fn_train, steps = 3000)


def loadDimensions(envName):
    import pickle
    data = pickle.load(open("./trainingData/"+envName+".p", "rb"))

    observations = data["observations"]
    actions = np.squeeze(data["actions"])
    Label_Dimension=actions.shape[1]
    Feature_Dimension=observations.shape[1]
    return Label_Dimension, Feature_Dimension


def loadEstimator(envName, Label_Dimension, Feature_Dimension):
    dirName = "./trained/"+envName+".p"
    print("Loading trianed model from",dirName)

    label_dimension, feature_dimension = loadDimensions(envName)

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

def applyDagger():
    #run episode of expert policy
    runEpisode(policy_fn, env, render = True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envName', type=str)
    parser.add_argument('numEpisodes', type=int)

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    estimator = loadEstimator(dirName, label_dimension, feature_dimension)

    import gym
    env = gym.make(args.envName)
    play(estimator, env, args.numEpisodes, True)

if __name__ == '__main__':
    main()
