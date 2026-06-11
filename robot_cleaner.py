from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

import torch
import omni.usd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation
writer = SummaryWriter("runs/robot_cleaner")

@dataclass
class Args:
    render_episodes:int = 10
    num_episodes:int = 800
    num_steps:int = 512
    batch_size:int = 32
    num_epochs:int = 4
    gamma:float = 0.99
    gae_lambda:float = 0.95
    clip_coef:float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 2.5e-4
    norm_adv:bool = True
    train_agent:bool = False

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space_num).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space_num).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64 ,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space_num), std=0.01)
        )

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, envs, action_indice=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action_indice is None:
            action_indice = dist.sample()
        action_space = torch.tensor(envs.action_space).to(device)    
        action = action_space[action_indice]
        log_prob = dist.log_prob(action_indice)
        entropy = dist.entropy()
        return action, action_indice, log_prob, entropy, self.critic(x)

class Env:
    def __init__(self):
        self.observation_space_num:int = 7
        self.action_space_num:int = 5
        self.steps:int = 1
        self.action_space = [[0,0], [0.8,0.8], [-0.8,-0.8],[0.8,-0.8],[-0.8,0.8]]
        self.terminate:bool = False
        self.truncate:bool = False
        self.collision:bool = False
        self.route_taken = []
        self.world = None
        self.robot = None

    def make(self):
        omni.usd.get_context().open_stage("robot_cleaner.usd")
        print("Simulation started!")
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/robot")
        print(f"Robot prim valid: {robot_prim.IsValid()}")
        for prim in stage.Traverse():
            if "joint" in prim.GetPath().pathString.lower():
                print(prim.GetPath())
        for i in range(100):
            app.update()
        self.world = World()
        self.robot = SingleArticulation("/World/robot")
        self.world.scene.add(self.robot)
        self.world.reset()
        self.robot.initialize()
        print(f"Joint names: {self.robot.dof_names}")

    def get_state(self):
        position, orientation = self.robot.get_world_pose()
        r = Rotation.from_quat([orientation[1],orientation[2], orientation[3],orientation[0]])
        yaw = r.as_euler('xyz')[2]
        left_wheel_vel, right_wheel_vel,_,_,_ = self.robot.get_joint_velocities()
        if [round(position[0],1), round(position[1],1)] not in self.route_taken:
            route_len = len(self.route_taken) + 1
        else:
            route_len = len(self.route_taken)
        coverage = round(route_len / self.steps,2)
        return [round(position[0],1), round(position[1],1), round(float(np.sin(yaw)),2), round(float(np.cos(yaw)),2), left_wheel_vel, right_wheel_vel, coverage]

    def check_truncate(self):
        if step == args.num_steps:
            self.truncate = True

    def check_collision(self, state):
        in_bottom = (2 < state[0] < 28) and (2 < state[1] < 10)
        in_top = (22 < state[0] < 28) and (14 < state[1] < 28)
        if not (in_bottom or in_top):
            self.collision = True
        else:
            self.collision = False

    def check_reward(self, state):
        reward = 0
        if self.collision:
            reward -= 5
            return reward
        if [state[0], state[1]] in self.route_taken:
            reward -= 0.5
        else:
            reward += 1
        reward += round(len(self.route_taken)/5400 * 0.01,2)
        return reward
    
    def reset(self):
        self.world.reset()
        self.terminate = False
        self.truncate = False
        self.route_taken = []
        self.steps = 1
        return self.get_state()
    
    def step(self, action):
        action = ArticulationAction(joint_velocities = action.squeeze().detach().cpu().numpy(), joint_indices=np.array([0, 1]) )    
        self.robot.apply_action(action)
        for _ in range(20):
            self.world.step(render=True)
        state = self.get_state()
        self.check_collision(state)
        self.check_truncate()
        reward = self.check_reward(state) 
        if [state[0], state[1]] not in self.route_taken:
            self.route_taken.append([state[0], state[1]])
        return state, reward

args = Args()

if __name__ == "__main__":
    if args.train_agent:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = Env()
        agent = Agent(env).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        episode_rewards = []
        env.make()
        obs = torch.zeros((args.num_steps, env.observation_space_num)).to(device)
        actions = torch.zeros((args.num_steps, 2)).to(device)
        action_indices = torch.zeros((args.num_steps,), dtype=torch.long).to(device)
        rewards = torch.zeros((args.num_steps,)).to(device)
        dones = torch.zeros((args.num_steps,)).to(device)
        values = torch.zeros((args.num_steps,)).to(device)
        logprobs = torch.zeros((args.num_steps,)).to(device)

        for episode in range(args.num_episodes):
            next_obs = env.reset()
            next_obs = torch.tensor(next_obs).to(device)
            next_done = torch.zeros(1).to(device)
            for step in range(args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, action_indice, logprob, _, value = agent.get_action_and_value(next_obs, env)
                    values[step] = value.squeeze()
                    actions[step] = action.squeeze()
                    action_indices[step] = action_indice.squeeze()
                    logprobs[step] = logprob.squeeze()
                env.steps += 1
                next_obs, reward = env.step(action)
                rewards[step] = torch.tensor(reward).to(device)
                next_obs, next_done = torch.tensor(next_obs).to(device), torch.tensor([float(env.terminate or env.truncate)], dtype=torch.float32).to(device)
                if env.terminate or env.truncate:
                    break
            with torch.no_grad():
                next_value = agent.get_value(next_obs)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(step)):
                    if t  == args.num_steps - 1:
                        nextnonterminate = 1 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminate = 1 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + args.gamma * nextnonterminate * nextvalues - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminate * lastgaelam
                returns = advantages + values
            episode_rewards.append(rewards[:step].sum().item())

            b_inds = np.arange(args.num_steps)
            for epoch in range(args.num_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, step, args.batch_size):
                    end = start + args.batch_size 
                    mb_inds = b_inds[start:end]
                    _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[mb_inds], env, action_indices[mb_inds])
                    logratio = newlogprob - logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean())/(mb_advantages.std() + 1e-8)
                    
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss = 0.5*((newvalue - returns[mb_inds])**2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                print("epoch update done")
            print(f"episode {episode} finished. final step: {step}. episode reward:{rewards[:step].sum().item()}")
            print(f"episode {episode}, pg_loss: {pg_loss.item():.4f}, v_loss: {v_loss.item():.4f}")
            writer.add_scalar("reward/episode", rewards[:step].sum().item(), episode)

        torch.save(agent.state_dict(), "robot_cleaner_ppo.pth")
        print("trained weights saved")

        fig, ax = plt.subplots()
        ax.plot(episode_rewards)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Robot Cleaner PPO Training")
        plt.show()
    else:
        all_episode_coords = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = Env()
        agent = Agent(env).to(device)
        agent.load_state_dict(torch.load("robot_cleaner_ppo.pth"))
        agent.eval() 
        env.make()
        rewards = torch.zeros((args.num_steps,)).to(device)
        for episode in range(args.render_episodes):
            next_obs = env.reset()
            next_obs = torch.tensor(next_obs).to(device)
            next_done = torch.zeros(1).to(device)
            for step in range(args.num_steps):
                with torch.no_grad():
                    action, action_indice, logprob, _, value = agent.get_action_and_value(next_obs, env)
                env.steps += 1
                next_obs, reward = env.step(action)
                rewards[step] = torch.tensor(reward).to(device)
                next_obs, next_done = torch.tensor(next_obs).to(device), torch.tensor([float(env.terminate or env.truncate)], dtype=torch.float32).to(device)
            print(f"episode {episode} finished. final step: {step}. episode reward:{rewards[:step].sum().item()}")
        print("rendering finished")






