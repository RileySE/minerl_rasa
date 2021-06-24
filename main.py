# Run minerl tests with simple RL agents
import minerl
import gym

import agents
import train

env = gym.make('MineRLNavigateDense-v0')

agent = agents.SimpleA2C(env, 'cuda:0').to('cuda:0')

train.train(agent, env, 10000)
