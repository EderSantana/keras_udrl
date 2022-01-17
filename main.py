import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# MODEL
def get_model(state_dim=4, cmd_dim=2, action_dim=2, hidden_dim=64):
    cmd_scale = .02
    state = keras.Input(shape=(state_dim,))
    cmd = keras.Input(shape=(cmd_dim,))

    x_emb = layers.Dense(hidden_dim, activation='sigmoid')(state)
    c_emb = layers.Dense(hidden_dim, activation='sigmoid')(cmd_scale * cmd)
    prod = layers.multiply([x_emb, c_emb])

    out = layers.Dense(hidden_dim, activation='relu')(prod)
    out = layers.Dense(hidden_dim, activation='relu')(out)
    out = layers.Dense(hidden_dim, activation='relu')(out)
    out = layers.Dense(action_dim, activation='softmax')(out)

    model = keras.Model(inputs=[state, cmd], outputs=out, name='UDRL')
    return model

def sample_action(probs, greedy=False):
    if greedy:
        action = np.argmax(probs)
    else: 
        action = np.argmax(np.random.multinomial(1, probs))
    return action

# EPISODE ROLLOUT / MODEL ACT IN THE ENV
def model_rollout(model, env, max_steps_per_ep=100, d_reward=100, t_horizon=100, buffr=None, silent=True, greedy=False):
    states = []
    actions = []
    rewards = []
    d = d_reward
    h = t_horizon
    state = env.reset()
    for step in range(max_steps_per_ep):
        if not silent:
            env.render()
        inputs = [state.reshape(1, -1), np.array([[d, h]])]
        action = model(inputs, training=False)[0]
        act = sample_action(action, greedy)
        next_state, reward, done, _ = env.step(act)
        states.append(state)
        rewards.append(reward)
        actions.append(act)

        state = next_state
        d = d-reward
        h = max(h-1.0, 1.0)
        if done:
            break
    if buffr is not None:
        buffr.append(states, actions, rewards)
    return sum(rewards)

# MEMORY / BUFFER / SAMPLING
class Buffer:
    def __init__(self, max_len=100):
       self.buffer = []
       self.max_len = max_len

    def append(self, states, actions, rewards):
        ep = {'states': states, 'actions': actions, 'rewards': rewards, 'sum_rewards': sum(rewards)}
        self.buffer.append(ep)

    def sort(self):
        self.buffer = sorted(self.buffer, key=lambda i: i["sum_rewards"], reverse=True)
        self.buffer = self.buffer[:self.max_len]

    def sample(self, n):
        self.sort()
        idxs = np.random.randint(0, len(self.buffer), n)
        samp = [self.buffer[i] for i in idxs]
        return samp

    def get_best(self, n):
        self.sort()
        return self.buffer[:n]

    def __len__(self):
        return len(self.buffer)

    def _prepare_inputs(self, ep,  t1, t2):
        state = ep["states"][t1]
        d_reward = np.array([sum(ep["rewards"][t1:t2])])  # get this many rewards...
        t_horizon = np.array([t2 - t1])  # ... in this ammount of time...
        action = ep["actions"][t1]  # ... starting with this actions
        return state, d_reward, t_horizon, action

    def sample_batch(self, batch_size):
        input_array = []
        output_array = []
        cmd_array = []
        episodes = self.sample(batch_size)
        for ep in episodes:
            T = len(ep["states"])
            t1 = np.random.randint(0, T-1)
            t2 = np.random.randint(t1+1, T)
            state, d_reward, t_horizon, action = self._prepare_inputs(ep, t1, t2)
            input_array.append(state)
            cmd_array.append(np.concatenate([d_reward, t_horizon]))
            output_array.append(action) 
        return [np.array(input_array), np.array(cmd_array)], np.array(output_array)

def get_cmd(buffr, nbest=50):
    best = buffr.get_best(nbest)
    horizon = np.mean([len(i["states"]) for i in best]) + 1
    rewards = [i["sum_rewards"] for i in best]
    r_mean = np.mean(rewards)
    r_stdv = np.std(rewards)
    reward = np.random.uniform(r_mean, r_mean+r_stdv)
    return reward, horizon
    
# MODEL TRAINING 
def training_loop(n_epochs, env, model, buffr):
    rewards = []
    for epoch in range(n_epochs):
        loss_array = []
        for i in range(batches_per_epoch):
            inputs, outputs = buffr.sample_batch(256)
            batch_loss = model.train_on_batch(inputs, outputs)
            loss_array.append(batch_loss)
        for j in range(exploration_per_epoch):
            d, h = get_cmd(buffr, 50)
            _ = model_rollout(model, env, d_reward=d, t_horizon=h, buffr=buffr, silent=True)

        d, h = get_cmd(buffr)
        eval_r = model_rollout(model, env, d_reward=d, t_horizon=h, silent=True, greedy=True)
        rewards.append(eval_r)
        print(f"Epoch {epoch} | Loss: {np.mean(loss_array)} | D: {d} | H: {h} | R: {eval_r}")
    return model, rewards

# params
seed = 42
n_replay = 600
n_warmup = 50
# training params
n_epochs = 100
batches_per_epoch = 100 
exploration_per_epoch = 15
batch_size = 256


# init
env = gym.make("CartPole-v0")
# env = gym.make("LunarLander-v2")
action_space = env.action_space
buffr = Buffer(n_replay)
env.seed(seed)
    
model = get_model(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
model.compile('adam', loss=keras.losses.SparseCategoricalCrossentropy()) 
model.summary()

# warmup
for j in range(50):
   model_rollout(model, env, d_reward=1, t_horizon=1, buffr=buffr, silent=True)

# learn
try:
    model, losses  = training_loop(200, env, model, buffr)
except KeyboardInterrupt:
    pass 

# evaluate 
R = []
for i in range(10):
    r = model_rollout(model, env, max_steps_per_ep=1000, d_reward=300, t_horizon=300, silent=False, greedy=True)
    R.append(r)
print(f"Avg Reward: {np.mean(R)}")
