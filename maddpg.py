# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F



class MADDPG:
    def __init__(self, seed, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPGAgent(24, 2,seed),
                             DDPGAgent(24, 2,seed)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = []
        for i in range(len(self.maddpg_agent)):
            action = self.maddpg_agent[i].act(obs_all_agents[i,:].view(1,-1),noise)
            actions.append(action.squeeze())
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        target_actions = []
        for i in range(len(self.maddpg_agent)):
            target_action = self.maddpg_agent[i].act(obs_all_agents[:, i,:],noise)
            target_actions.append(target_action)
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        # state, cat_state, actions, rewards, next_state, cat_next_state, dones)

        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples
        batch_size = obs_full.shape[0]
        # obs_full = torch.stack(obs_full)
        # next_obs_full = torch.stack(next_obs_full)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network

        target_actions = self.target_act(next_obs.view(batch_size, 2, -1))
        target_actions = torch.cat(target_actions, dim=1)

        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full,target_actions.to(device))
        y = reward[:,agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[:,agent_number].view(-1, 1))

        q = agent.critic(obs_full,action.view(batch_size,-1))

        # huber_loss = torch.nn.SmoothL1Loss()
        # critic_loss = huber_loss(q, y.detach())
        critic_loss = F.mse_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(obs.view(batch_size,2,-1)[:,i,:]) if i == agent_number \
                   else self.maddpg_agent[i].actor(obs.view(batch_size,2,-1)[:,i,:]).detach()
                   for i in range(2) ]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        # q_input2 = torch.cat((obs_full.t(), q_input), dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(obs_full,q_input).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)
        self.update_targets()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
