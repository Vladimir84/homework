import tensorflow as tf
import numpy as np
import gym
import load_policy
from EvaluatePolicy import evaluate

def runEpisode(policy_fn, env, render = True):
    observations = []
    actions = []
    done = False
    totalr = 0.
    steps = 0
    max_steps = env.spec.timestep_limit

    obs = env.reset()
    while not done:
        action = policy_fn(obs[None,:])
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if len(actions) % 10 == 0 and render:
           env.render()
        if len(actions)>=max_steps:
           break

    return observations, actions

def policyConverter(estimator):
    def prediction(obs):
        def input_fn_train(): # returns x, Y       
            return {"": tf.constant(np.array(obs), dtype=tf.float32)}
        return estimator.predict(input_fn=input_fn_train,as_iterable=False)
    return prediction

def askExpert(expertPolicy, observations):
    expertActions = []
    for observation in observations:
        action = expertPolicy(observation[None,:])
        expertActions.append(action)
    return expertActions

def trainImitator(estimator, observations, actions):
    def input_fn_train(): # returns x, Y       
        return {"": tf.constant(np.array(observations), dtype=tf.float32)}, \
               tf.constant(np.squeeze(np.array(actions)), dtype=tf.float32)

    estimator.fit(input_fn=input_fn_train, steps = 3000)
#    return estimator

#def loadDimensions(env):
#
#    # observation_sample = env.observation_space.sample() # bug in mtrand.RandomState.uniform
#    action_sample = env.action_space.sample()
#
#    feature_dimension = observation_sample.shape[0]
#    label_dimension = action_sample.shape[0]
#
#    return feature_dimension, label_dimension

def createImitator(feature_dimension, label_dimension, dir_name):
    feature_column = [tf.contrib.layers.real_valued_column("", dimension=feature_dimension)]
    estimator = tf.contrib.learn.DNNRegressor(
                   feature_columns=feature_column,
                   hidden_units=[64,64],
                   #activation_fn = tf.nn.relu,
                   activation_fn = tf.tanh,
                   model_dir = dir_name,
                   label_dimension=label_dimension
                )
    return estimator

def applyDagger(expert_policy, imitator, env, num_iterations=50):
    imitation_policy = policyConverter(imitator)

    means = []
    stds = []
    with tf.Session():
       #run episode of expert policy
       observations, actions = runEpisode(expert_policy, env)

       for iteration in range(num_iterations):
          #train imitation policy
          trainImitator(imitator, observations, actions)
          #run imitation_policy
          observations_episode, _ = runEpisode(imitation_policy, env)
          expert_actions =  askExpert(expert_policy, observations_episode)
          observations.extend(observations_episode)
          actions.extend(expert_actions)

          mean, std = evaluate(env, imitation_policy)
          means.append(mean)
          stds.append(std)

    return means, stds


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('num_iterations', type=int)
    #because motherfucking bugs 
    parser.add_argument('feature_dimension', type=int)
    parser.add_argument('label_dimension', type=int)
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    expert_policy = load_policy.load_policy("experts/"+args.env_name+".pkl")
    dir_name = "trained/"+args.env_name+".dagger"
    imitator = createImitator(args.feature_dimension, args.label_dimension, dir_name)

    env = gym.make(args.env_name)

    means, stds = applyDagger(expert_policy, imitator, env, args.num_iterations)

    data = {'mean_returns':means,
            'std_returns':stds}
    pickle.dump(data,open("dagger/"+args.env_name+".dat") )


if __name__ == '__main__':
    main()
