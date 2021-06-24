# Misc utility functions
import torch


# Convert data dicts to pytorch
def convert_obs(obs, device='cpu'):
    pov = torch.as_tensor(obs['pov'], device=device, dtype=torch.float32)
    comp = torch.as_tensor(obs['compassAngle'], device=device, dtype=torch.float32).unsqueeze(0)
    inv = torch.as_tensor(obs['inventory']['dirt'], device=device, dtype=torch.float32).unsqueeze(0)

    # Rescale RBG values
    pov /= 255.

    return pov, comp, inv


# Convert the given action as a OrderedDict into a pytorch vector
def convert_action_dict_to_vec(act_dict):
    act_vec = torch.zeros(11)
    act_vec[0] = act_dict['attack']
    act_vec[1] = act_dict['back']
    act_vec[2] = act_dict['camera'][0]
    act_vec[3] = act_dict['camera'][1]
    act_vec[4] = act_dict['forward']
    act_vec[5] = act_dict['jump']
    act_vec[6] = act_dict['left']
    # TODO handle place being a string instead of an int
    act_vec[7] = float(act_dict['place'])
    act_vec[8] = act_dict['right']
    act_vec[9] = act_dict['sneak']
    act_vec[10] = act_dict['sprint']
    return act_vec


# Grab dists given logits
def get_dists(act_logits):

    # Binary variables
    attack_p = act_logits[:, 0]
    back_p = act_logits[:, 1]
    forward_p = act_logits[:, 4]
    jump_p = act_logits[:, 5]
    left_p = act_logits[:, 6]
    right_p = act_logits[:, 8]
    sneak_p = act_logits[:, 9]
    sprint_p = act_logits[:, 10]

    # For these, let's use a bernoulli dist
    dist = torch.distributions.bernoulli.Bernoulli
    attack_dist = dist(logits=attack_p)
    back_dist = dist(logits=back_p)
    forward_dist = dist(logits=forward_p)
    jump_dist = dist(logits=jump_p)
    left_dist = dist(logits=left_p)
    right_dist = dist(logits=right_p)
    sneak_dist = dist(logits=sneak_p)
    sprint_dist = dist(logits=sprint_p)

    # Camera vars (x and y rotation, euler angles?)
    camera_x_dist = torch.distributions.normal.Normal(act_logits[:, 2], scale=0.1)
    camera_y_dist = torch.distributions.normal.Normal(act_logits[:, 3], scale=0.1)

    # placement enum, currently only dirt versus nothing supported
    # TODO at some point this should be softmax with more vars
    place_p = act_logits[:, 7]
    # This one can also be a bernoulli dist for now
    place_dist = dist(logits=place_p)

    return attack_dist, back_dist, camera_x_dist, camera_y_dist, forward_dist, jump_dist, left_dist, place_dist, \
        right_dist, sneak_dist, sprint_dist
