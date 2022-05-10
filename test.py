from __future__ import print_function
from array import array

import sys

sys.path.append("../")

from datetime import datetime
import numpy as np
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, hl, rendering=True, max_timesteps=2000):
    episode_reward = 0
    step = 0

    state = env.reset()

    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events()
    counter = 0
    while True:
        state = image_processing(state)
        if step == 0:
            state_1, state_2, state_3, state_4 = state, state, state, state

        hist_5 = np.stack((state_4, state_3, state_2, state_1, state))
        hist_3 = np.stack((state_2, state_1, state))

        if hl == 1:
            hist = state.reshape(1, 96, 96)
        elif hl == 3:
            hist = hist_3
        elif hl == 5:
            hist = hist_5

        a = agent.predict(hist).detach().cpu().numpy().argmax(1)
        ##################################################################
        # OVER-WRITING AGENT DUE TO PREVENTING AGENT BEING STUCK SOMETIMES
        if np.array_equal(state, state_1):
            counter = 15
            print("Agent got stuck, initial push is given to continue")
        if counter > 0:
            a = 3
            counter -= 1
        ###################################################################
        a = id_to_action(a)
        next_state, r, done, info = env.step(a)
        if step > 1:
            state_4 = state_3
            state_3 = state_2
            state_2 = state_1
            state_1 = state
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15  # number of episodes to test

    hl = 5
    agent = BCAgent(history_length=hl, lr=1e-4)
    agent.load("models/agent_5.pt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, hl, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
