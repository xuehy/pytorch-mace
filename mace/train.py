from torch.autograd import Variable
from madrl_environments.pursuit import MAWaterWorld_mod_mixed
from model import NeuralCoord
import numpy as np
import torch as th
from params import args
import torchvision.transforms as T
from torch.optim import Adam
import torch.nn.functional as F


epsilon = 1e-15
resize = T.Compose([
    T.ToPILImage(),
    T.Scale(84, interpolation=T.Image.CUBIC),
    T.ToTensor()])


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, shared_model, optimizer=None):
    max_episode_length = args.max_episode_length
    num_steps = args.num_steps
    th.manual_seed(args.seed + rank)
    world = MAWaterWorld_mod_mixed(n_pursuers=args.n_pursuers,
                                   n_evaders=args.n_evaders,
                                   n_poison=args.n_poison,
                                   obstacle_radius=0.04,
                                   radius=args.radius,
                                   food_reward=args.food_reward,
                                   poison_reward=args.poison_reward,
                                   encounter_reward=args.encounter_reward,
                                   n_coop=args.n_coop,
                                   sensor_range=1.0,
                                   ev_speed=0.02,
                                   poison_speed=0.02,
                                   obstacle_loc=None, )
    world.seed(args.seed + rank)
    mace = NeuralCoord(args.n_pursuers,
                       args.n_states,
                       args.dim_action)
    if mace.use_cuda:
        mace.cuda()
    if optimizer is None:
        optimizer = Adam(mace.parameters(), lr=0.0001)
    FloatTensor = th.cuda.FloatTensor if mace.use_cuda else th.FloatTensor
    mace.train()

    state = world.reset()
    state = th.from_numpy(
        np.concatenate(state)).float()
    img = world.render()
    img = resize(img)
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        mace.load_state_dict(shared_model.state_dict())
        values = []
        log_probs = []
        rewards = []
        KL_divergence = []
        entropies = []
        for step in range(num_steps):
            v, p, lmu, llogvar, mu, logvar = mace(
                Variable(state.unsqueeze(0)), Variable(
                    img.unsqueeze(0)))
            prob = F.softmax(p)
            log_prob = F.log_softmax(p + epsilon)

            entropy = -(log_prob * prob).sum(1)

            entropies.append(entropy)

            # action = prob.multinomial().data
            action = prob.max(1)[1].data
            log_prob = log_prob.gather(1, Variable(action))
            state, reward, done, _, _ = world.step(action.squeeze().numpy())
            state = th.from_numpy(np.concatenate(state)).float()
            img = world.render()
            img = resize(img)
            done = done or episode_length >= max_episode_length
            reward = th.FloatTensor(reward).type(FloatTensor)

            if done:
                episode_length = 0
                state = world.reset()
                state = th.from_numpy(
                    np.concatenate(state)).float()
                img = resize(world.render())

            values.append(v)
            log_probs.append(log_prob)
            rewards.append(reward*args.scale_reward)
            KL_loss = lmu.pow(2).mul_(1.0).add_(
                llogvar.exp()).mul_(-1).add_(1).add_(llogvar)
            KL_loss = th.sum(KL_loss).mul_(-0.5)
            kl = []
            for a in range(args.n_pursuers):
                kl_z = mu[a].pow(2).add_(
                    logvar[a].exp()).mul_(-1).add_(1).add_(logvar[a])
                kl.append(th.sum(kl_z).mul_(-0.5))
            kl = th.cat(kl).sum()
            KL_divergence.append(kl + KL_loss)
            if done:
                break

        R = th.zeros(args.n_pursuers)
        if not done:
            v, _, _, _, _, _ = mace(Variable(state.unsqueeze(0)),
                                    Variable(img.unsqueeze(0)))
            R = v.data
        values.append(Variable(R))

        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = th.zeros(args.n_pursuers)
        for i in reversed(range(len(rewards))):
            R = Variable(th.Tensor(rewards[i])) + (
                args.gamma * R)
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2).sum()
            delta_t = rewards[i] + args.gamma * (
                values[i + 1].data) - values[i].data
            gae = gae * args.gamma + delta_t
            policy_loss = policy_loss - (
                log_probs[i] * Variable(gae)).sum() + \
                KL_divergence[i] - args.beta * entropies[i].sum()

        optimizer.zero_grad()

        (policy_loss + value_loss * 0.5).backward()
        th.nn.utils.clip_grad_norm(mace.parameters(), 1)

        ensure_shared_grads(mace, shared_model)
        optimizer.step()
