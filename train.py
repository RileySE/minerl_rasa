import gym


def train(agent, env, n_epis):

    for epi in range(n_epis):
        obs = env.reset()
        net_reward = 0
        done = False
        while not done:
            action = agent.sample_action(obs)

            obs, reward, done, info = env.step(
                action)

            net_reward += reward
            agent.update(obs, action, reward, done)

        print("Total reward: ", net_reward)
