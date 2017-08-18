from collections import namedtuple

Arg = namedtuple('Arg', [
    'n_pursuers',
    'n_evaders',
    'n_poison',
    'food_reward',
    'poison_reward',
    'encounter_reward',
    'n_coop',
    'seed',
    'scale_reward',
    'n_states',
    'gamma',
    'dim_action',
    'num_process',
    'lr',
    'beta',
    'max_episode_length',
    'num_steps',
    'port',
    'radius',
    'latent_dim',
    'test_interval'
])

args = Arg(
    scale_reward=0.01,
    n_pursuers=2,
    n_evaders=3,
    n_poison=3,
    food_reward=10,
    poison_reward=-1,
    encounter_reward=0.01,
    n_coop=2,
    seed=1234,
    n_states=183,
    gamma=0.99,
    dim_action=4,
    num_process=12,
    lr=1e-6,
    beta=0.1,  # weight for the entropy
    max_episode_length=1000,
    num_steps=20,
    port=5275,
    latent_dim=20,
    test_interval=60,
    radius=0.05)

visual_input = False
