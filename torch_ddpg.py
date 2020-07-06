from ddpg_torch import Agent
import gym
import numpy as np
#from utils import PlotLearning
env = gym.make('LunarLanderContinuous-v2')
agent = Agent(alpha=0.000025,beta=0.00025, input_dims=[8],tau=0.001,env=env)
np.random.seed(0)
agent.load_models()
score_history = []
while True:
    done = False
    score = 0
    obs = env.reset()
    while not done:
        env.render()
        act = agent.choose_action(obs)
        new_state,reward,done,info = env.step(act)
        agent.remember(obs,act,reward,new_state,int(done))
        #agent.learn()
        score += reward
        obs = new_state
    score_history.append(score)
    print('episode',i,'score%.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))
    if i % 25 == 0:
        agent.save_models()    
    if len(score_history) >= 25 and np.mean(score_history[-25:]) > 200:
        break
filename = 'pendulum.png'
#plotLearning(score_history, filename, window=100)