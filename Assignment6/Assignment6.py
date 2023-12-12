# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:39:22 2023

@author: Kugavathanan
"""

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver

# Set up the environment
env_name = 'LunarLander-v2'
train_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

# Define the Q-network
fc_layer_params = (100,)

# Rationale for the choice of activation functions and loss function:
# Activation Function: ReLU is used for hidden layers, and no activation function for the output layer.
# Loss Function: Mean squared error loss is used as the TD error loss for DQNs.

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

train_step_counter = tf.Variable(0)

# Define the DQN agent
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

# Set up replay buffer and observer
replay_buffer_capacity = 100000

# Strategy for adjusting hyperparameters:
# You can use a hyperparameter tuning library like Optuna or implement a manual grid search.
# Adjust hyperparameters like the number of iterations, number of episodes, maximum number of steps, and discount factor.

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

replay_buffer_observer = replay_buffer.add_batch

# Define training metrics
average_return = tf_metrics.AverageReturnMetric()

# Define the driver
collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + [average_return],
    num_steps=1)

# Define the dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

iterator = iter(dataset)

# Training loop
num_iterations = 10000
for iteration in range(num_iterations):
    # Collect a step using the collect policy
    time_step = train_env.current_time_step()
    action_step = agent.collect_policy.action(time_step)
    next_time_step = train_env.step(action_step.action)

    # Add the step to the replay buffer
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)

    # Train the agent
    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss

    # Log metrics
    if iteration % 1000 == 0:
        print('Iteration: {}, Loss: {}'.format(iteration, train_loss))

# Evaluate the agent
num_eval_episodes = 10
for _ in range(num_eval_episodes):
    time_step = eval_env.reset()
    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)

# Plotting metrics
import matplotlib.pyplot as plt

returns = [step[0].numpy() for step in average_return.result()]
plt.plot(returns)
plt.ylabel('Average Return')
plt.xlabel('Iteration')
plt.show()
