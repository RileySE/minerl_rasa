import gym
import wandb


def train(agent, env, n_epis):

    for epi in range(n_epis):
        obs = env.reset()
        net_reward = 0
        done = False
        epi_act_loss = 0
        epi_value_loss = 0
        n_steps = 0
        while not done:
            n_steps += 1
            action = agent.sample_action(obs)

            next_obs, reward, done, info = env.step(
                action)

            net_reward += reward
            act_loss, value_loss, entropy = agent.update(obs, action, reward, done, next_obs)
            epi_act_loss += act_loss
            epi_value_loss += value_loss
            obs = next_obs

        wandb.log({'action_loss': epi_act_loss, 'value_loss': epi_value_loss, 'return': net_reward,
                   'episode_length': n_steps, 'entropy': entropy})
        print("Total reward: ", net_reward)
