from torch.autograd import Variable
from madrl_environments.pursuit import MAWaterWorld_mod_mixed
from model import NeuralCoord
import numpy as np
import torch as th
import visdom
import torchvision.transforms as T
from params import args
import time
import torch.nn.functional as F


resize = T.Compose([
    T.ToPILImage(),
    T.Scale(84, interpolation=T.Image.CUBIC),
    T.ToTensor()])


def test(rank, shared_model):
    vis = visdom.Visdom(port=args.port)
    win = None
    win1 = None
    max_episode_length = 900
    th.manual_seed(args.seed + rank)
    world = MAWaterWorld_mod_mixed(n_pursuers=args.n_pursuers,
                                   n_evaders=args.n_evaders,
                                   n_poison=args.n_poison,
                                   obstacle_radius=0.04,
                                   food_reward=args.food_reward,
                                   poison_reward=args.poison_reward,
                                   encounter_reward=args.encounter_reward,
                                   n_coop=args.n_coop,
                                   sensor_range=1.0,
                                   radius=args.radius,
                                   ev_speed=0.02,
                                   poison_speed=0.02,
                                   obstacle_loc=None, )
    world.seed(args.seed + rank)
    mace = NeuralCoord(args.n_pursuers,
                       args.n_states,
                       args.dim_action)
    if mace.use_cuda:
        mace.cuda()

    mace.eval()

    state = world.reset()
    state = th.from_numpy(
        np.concatenate(state)).float()
    img = resize(world.render())
    done = True
    episode_length = 0
    total_reward = 0
    agent_reward = np.zeros(args.n_pursuers)
    logvar_sum = 0
    lvar = 0
    entropy = 0
    episode = 0
    KL_loss = 0
    catch = 0
    while True:
        episode_length += 1
        if done:
            mace.load_state_dict(shared_model.state_dict())
        v, p, lmu, llogvar, mu, logvar = mace(
            Variable(state.unsqueeze(0)),
            Variable(img.unsqueeze(0)))
        prob = F.softmax(p)
        log_prob = F.log_softmax(p)
        entropy_ = -(prob * log_prob).sum()
        entropy += entropy_.data[0]
        # action = prob.multinomial().data
        action = prob.max(1)[1].data
        # directly take mu as the action w/o Gaussian
        state, reward, done, _, _ = world.step(action.numpy())
        state = th.from_numpy(np.concatenate(state)).float()
        img = resize(world.render())
        done = done or episode_length >= max_episode_length

        total_reward += reward.sum()
        agent_reward += reward
        lvar += llogvar.exp().sum().data[0]
        logvar_sum += logvar.exp().sum().data[0]

        KL_loss_ = lmu.pow(2).add_(
            llogvar.exp()).mul_(-1).add_(1).add_(llogvar)
        KL_loss_ = th.sum(KL_loss_).mul_(-0.5)
        kl = []
        for a in range(args.n_pursuers):
            kl_z = mu[a].pow(2).add_(
                logvar[a].exp()).mul_(-1).add_(1).add_(logvar[a])
            kl.append(th.sum(kl_z).mul_(-0.5))
        kl = th.cat(kl).sum()
        KL_loss += kl.data[0] + KL_loss_.data[0]
        if done:
            print('MACE on WaterWorld\n' +
                  'scale_reward=%f\n' % args.scale_reward +
                  'agent=%d' % args.n_pursuers +
                  ', coop=%d' % args.n_coop +
                  ' food=%f, poison=%f, encounter=%f' % (
                      args.food_reward,
                      args.poison_reward,
                      args.encounter_reward))

            print('total rewrd = %f' % total_reward)
            print('z var = %f' % (logvar_sum / max_episode_length
                                  / args.latent_dim / 2))
            print('lambda var = %f' % (lvar / max_episode_length /
                                       args.latent_dim))
            print('entropy = %f' % (entropy / max_episode_length / 2))
            print('KL div = %f' % (KL_loss / max_episode_length / 2))
            print('----')
            if win is None:
                win = vis.line(X=np.arange(episode, episode+1),
                               Y=np.array([
                                   np.append(total_reward,
                                             agent_reward)]),
                               opts=dict(
                                   ylabel='Average Reward',
                                   xlabel='Iter(min)',
                                   title='MACE on WaterWorld_mod\n',
                                   legend=['Total'] +
                                   ['Agent-%d' % i for
                                    i in range(args.n_pursuers)]))
            else:
                vis.line(X=np.array(
                    [np.array(episode).repeat(args.n_pursuers+1)]),
                         Y=np.array([np.append(total_reward,
                                               agent_reward)]),
                         win=win,
                         update='append')

            if win1 is None:
                win1 = vis.line(X=np.arange(episode, episode+1),
                                Y=np.array([KL_loss / max_episode_length / 2]),
                                opts=dict(
                                    ylabel='KL Divergence',
                                    xlabel='Iter',
                                    title='MACE on WaterWorld_mod\n'))
            else:
                vis.line(X=np.array(
                    [episode]),
                         Y=np.array([KL_loss / max_episode_length / 2]),
                         win=win1,
                         update='append')

            total_reward = 0
            episode_length = 0
            KL_loss = 0
            entropy = 0
            logvar = 0
            logvar_sum = 0
            lvar = 0
            catch = 0
            state = world.reset()
            state = th.from_numpy(np.concatenate(state)).float()
            img = resize(world.render())
            agent_reward = np.zeros(args.n_pursuers)

            time.sleep(args.test_interval)
            episode += 1
