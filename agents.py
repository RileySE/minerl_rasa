# RL agents for minerl env, just playing around for now
from collections import OrderedDict

import torch
import torch.nn as nn


# Simple policy gradient agent
from utils import convert_obs, convert_action_dict_to_vec, get_dists


class SimpleA2C(nn.Module):
    def __init__(self, env, device='cpu'):
        super(SimpleA2C, self).__init__()

        # Assume obs consists of pov, compass, and inventory
        self.pov_shape = env.observation_space['pov'].shape  # Should be (64, 64, 3)
        self.inventory_shape = 1  # This is a single scalar
        self.compass_shape = 1  # Scalar between -180 and 180
        self.action_template = env.action_space.noop()

        self.conv_hidden_size = 64
        self.gamma = 0.99
        self.device = device

        # Make network layers
        # Conv down to 1 pixel
        self.conv_1 = nn.Conv2d(3, self.conv_hidden_size, 5, stride=2, padding=2)
        self.conv_2 = nn.Conv2d(self.conv_hidden_size, self.conv_hidden_size, 5, stride=2, padding=2)
        self.conv_3 = nn.Conv2d(self.conv_hidden_size, self.conv_hidden_size, 5, stride=2, padding=2)
        self.conv_4 = nn.Conv2d(self.conv_hidden_size, self.conv_hidden_size, 8)
        # FC layers
        # Input is self.conv_hidden_size values from conv stack plus compass and inventory scalars
        self.fc_1 = nn.Linear(self.conv_hidden_size + 2, 256)
        self.fc_2 = nn.Linear(256, 256)
        # 11 output values, mostly parametrize distributions for different action components
        self.fc_out = nn.Linear(256, 11)
        # Value output head
        self.value_out = nn.Linear(256, 1)

        # Track stuff for updates
        # Hopefully 6000 is enough here
        self.buffer_max = 8
        # Obs buffer is +1 bigger to store next obs for value bootstrapping
        self.epi_obs_pov = torch.zeros((self.buffer_max+1,) + self.pov_shape, device=self.device)
        self.epi_obs_inv = torch.zeros(self.buffer_max+1, self.inventory_shape, device=self.device)
        self.epi_obs_comp = torch.zeros(self.buffer_max+1, self.inventory_shape, device=self.device)
        self.epi_rew = torch.zeros(self.buffer_max, 1, device=self.device)
        self.epi_acts = torch.zeros(self.buffer_max, 11, device=self.device)
        self.epi_ind = 0

        self.opt = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, pov, comp, inv):
        # Move channel dim
        pov = pov.permute(0, 3, 1, 2)

        x = nn.functional.relu(self.conv_1(pov))
        x = nn.functional.relu(self.conv_2(x))
        x = nn.functional.relu(self.conv_3(x))
        x = nn.functional.relu(self.conv_4(x))

        fc_in = torch.cat([x.view(pov.size(0), -1), comp, inv], 1)
        y = nn.functional.relu(self.fc_1(fc_in))
        y = nn.functional.relu(self.fc_2(y))
        out = self.fc_out(y)
        value = self.value_out(y)

        return out, value

    # Sample an action... slightly complicated
    def sample_action(self, obs):
        # Get output and split into different distributions
        pov, comp, inv = convert_obs(obs, self.device)
        act_logits = self.forward(pov.unsqueeze(0), comp.unsqueeze(0), inv.unsqueeze(0))[0]

        attack_dist, back_dist, camera_x_dist, camera_y_dist, forward_dist, jump_dist, left_dist, place_dist,\
            right_dist, sneak_dist, sprint_dist = get_dists(act_logits)

        camera_x = camera_x_dist.sample()
        camera_y = camera_y_dist.sample()

        # TODO handle wrap around
        camera_x = (camera_x * 22.5).clamp(-180, 180)
        camera_y = (camera_y * 22.5).clamp(-180, 180)

        # Assemble the action dict
        action_dict = self.action_template
        action_dict['attack'] = attack_dist.sample()
        action_dict['back'] = back_dist.sample()
        action_dict['camera'] = [camera_x.item(), camera_y.item()]
        action_dict['forward'] = forward_dist.sample()
        action_dict['jump'] = jump_dist.sample()
        action_dict['left'] = left_dist.sample()
        action_dict['place'] = int(place_dist.sample())
        action_dict['right'] = right_dist.sample()
        action_dict['sneak'] = sneak_dist.sample()
        action_dict['sprint'] = sprint_dist.sample()
        return action_dict

    # Update agent
    def update(self, obs, action, reward, done, next_obs):

        pov, comp, inv = convert_obs(obs, self.device)
        self.epi_obs_pov[self.epi_ind] = pov
        self.epi_obs_inv[self.epi_ind] = comp
        self.epi_obs_comp[self.epi_ind] = inv
        self.epi_rew[self.epi_ind] = reward
        self.epi_acts[self.epi_ind] = convert_action_dict_to_vec(action)
        self.epi_ind += 1
        # Store next obs
        pov, comp, inv = convert_obs(next_obs, self.device)
        self.epi_obs_pov[self.epi_ind] = pov
        self.epi_obs_inv[self.epi_ind] = comp
        self.epi_obs_comp[self.epi_ind] = inv

        total_loss = 0
        value_loss = 0
        entropy = 0
        # Actually perform an update
        if done or self.epi_ind == self.buffer_max:

            # Get dists and compute log probs
            act_logits, values = self.forward(self.epi_obs_pov[:self.epi_ind+1], self.epi_obs_comp[:self.epi_ind+1],
                                              self.epi_obs_inv[:self.epi_ind+1])
            # Don't use logits for final observation (just for bootstrapping)
            act_logits = act_logits[:self.epi_ind]

            # Accumulate rewards
            returns = torch.zeros_like(self.epi_rew)
            for r in range(self.epi_ind - 1, 0, -1):
                returns[r] = self.epi_rew[r] + self.gamma * 0.95 * returns[min(r+1, self.buffer_max-1)] + \
                            (self.gamma * values[r+1].detach() - values[r])

            dists = get_dists(act_logits)
            advantages = returns[:self.epi_ind] # - values

            # Compute updates for each component of the action separately
            self.opt.zero_grad()
            for dist_n in range(len(dists)):
                log_probs = dists[dist_n].log_prob(self.epi_acts[:self.epi_ind, dist_n])
                entropy += dists[dist_n].entropy().mean()
                losses = -log_probs * advantages.detach()
                loss = losses.mean()
                total_loss += loss
                #loss.backward(retain_graph=True)

            total_loss /= len(dists)
            entropy /= len(dists)

            value_loss = advantages.pow(2).mean()

            (total_loss + value_loss + 0.01 * -entropy).backward()
            self.opt.step()

            # Cleanup
            self.epi_ind = 0
            self.epi_obs_pov = torch.zeros_like(self.epi_obs_pov)
            self.epi_obs_comp = torch.zeros_like(self.epi_obs_comp)
            self.epi_obs_inv = torch.zeros_like(self.epi_obs_inv)
            self.epi_rew = torch.zeros_like(self.epi_rew)
            self.epi_acts = torch.zeros_like(self.epi_acts)

        return total_loss, value_loss, entropy
