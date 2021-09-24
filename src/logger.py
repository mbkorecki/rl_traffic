import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import random
import pickle
import dill

class Logger:
    """
    The Logger class is responsible for logging data, building representations and saving them in a specified location
    """
    def __init__(self, args):
        """
        Initialises the logger object
        :param args: the arguments passed by the user
        """

        self.args = args
        
        self.veh_count = []
        self.travel_time = []
        self.losses = []
        self.plot_rewards = []
        self.episode_losses = []

        self.reward = 0

        self.log_path = "../" + args.sim_config.split('/')[2] +'_' + 'config' + args.sim_config.split('/')[3].split('.')[0] + '_' + str(args.agents_type)

        if args.load != None:
            self.log_path += "_load"
            
        
        old_path = self.log_path
        i = 1

        while os.path.exists(self.log_path):
            self.log_path = old_path + "(" + str(i) + ")"
            i += 1

        os.mkdir(self.log_path)


    def log_measures(self, environ):
        """
        Logs measures such as reward, vehicle count, average travel time and q losses, works for learning agents and aggregates over episodes
        :param environ: the environment in which the model was run
        """
        self.reward = 0
        for agent in environ.agents:
            self.reward += (agent.total_rewards / agent.reward_count)

        self.plot_rewards.append(self.reward)
        self.veh_count.append(environ.eng.get_finished_vehicle_count())
        self.travel_time.append(environ.eng.get_average_travel_time())
        self.episode_losses.append(np.mean(self.losses))
        
    def serialise_data(self, environ):
        """
        Serialises the waiting times data and rewards for the agents as dictionaries with agent ID as a key
        """
        waiting_time_dict = {}
        reward_dict = {}
        
        for agent in environ.agents:
            waiting_time_dict.update({agent.ID : {}})
            reward_dict.update({agent.ID : agent.total_rewards / agent.reward_count})
            for move in agent.movements.values():
                waiting_time_dict[agent.ID].update({move.ID : (move.max_waiting_time, move.waiting_time_list)})


        with open(self.log_path + "/" + "memory.dill", "wb") as f:
            dill.dump(environ.memory.memory, f)

        with open(self.log_path + "/" + "waiting_time.pickle", "wb") as f:
            pickle.dump(waiting_time_dict, f)
            
        with open(self.log_path + "/" + "agents_rewards.pickle", "wb") as f:
            pickle.dump(reward_dict, f) 


        if environ.agents_type == 'learning' or environ.agents_type == 'hybrid' or environ.agents_type == 'presslight' or environ.agents_type == 'policy':
            with open(self.log_path + "/" + "episode_rewards.pickle", "wb") as f:
                pickle.dump(self.plot_rewards, f)

            with open(self.log_path + "/" + "episode_veh_count.pickle", "wb") as f:
                pickle.dump(self.veh_count, f)

            with open(self.log_path + "/" + "episode_travel_time.pickle", "wb") as f:
                pickle.dump(self.travel_time, f)
            
    def save_log_file(self, environ):
        """
        Creates and saves a log file with information about the experiment in a .txt format
        :param environ: the environment in which the model was run
        """
        log_file = open(self.log_path + "/logs.txt","w+")

        log_file.write(str(self.args.sim_config))
        log_file.write("\n")
        log_file.write(str(self.args.num_episodes))
        log_file.write("\n")
        log_file.write(str(self.args.num_sim_steps))
        log_file.write("\n")
        log_file.write(str(self.args.update_freq))
        log_file.write("\n")
        log_file.write(str(self.args.batch_size))
        log_file.write("\n")
        log_file.write(str(self.args.lr))
        log_file.write("\n")
        
        log_file.write("mean vehicle count: " + str(np.mean(self.veh_count[self.args.num_episodes-10:])) + " with sd: " + str(np.std(self.veh_count[self.args.num_episodes-10:])) +
                       "\nmean travel time: " + str(np.mean(self.travel_time[self.args.num_episodes-10:])) +
                       " with sd: " + str(np.std(self.travel_time[self.args.num_episodes-10:])) +
                       "\nmax vehicle time: " + str(np.max(self.veh_count)) +
                       "\nmin travel time: " + str(np.min(self.travel_time))
                       )
        log_file.write("\n")
        log_file.write("best epoch: " + str(environ.best_epoch))
        log_file.write("\n")
        log_file.write("\n")

        for agent in environ.agents:
            log_file.write(agent.ID + "\n")
            for move in agent.movements.values():
                log_file.write("movement " + str(move.ID) + " max wait time: " + str(move.max_waiting_time) + "\n")
                if not move.waiting_time_list: move.waiting_time_list = [0]
                log_file.write("movement " + str(move.ID) + " avg wait time: " + str(np.mean(move.waiting_time_list)) + "\n")
            log_file.write("\n")
    
        log_file.write("\n")
        
        log_file.close()

    def save_phase_plots(self, environ):
        for agent in random.sample(environ.agents, 5):
            plt.plot(agent.past_phases, '|', linewidth=25)
            figure = plt.gcf()
            figure.set_size_inches(20,10)
            # plt.xticks(np.arange(0, self.args.num_sim_steps+1, step=10))
            plt.ylabel('phase')
            plt.xlabel('time')
            plt.grid()
            
            ax = plt.gcf().get_axes()[0]
            ax.spines['left'].set_position(('data', 0))
            ax.spines['bottom'].set_position(('data', 0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_xticks(np.arange(0, self.args.num_sim_steps+1, step=10), minor=True)
            ax.set_xticks(np.arange(0, self.args.num_sim_steps+1, step=100))
            ax.set_yticks(np.arange(-1, 8, step=1))
            ax.grid(which='minor', axis='both')
            
            plt.savefig(self.log_path + '/phase' + str(agent.ID) + '.png', bbox_inches='tight')
            plt.clf()


    def save_measures_plots(self):
        """
        Saves plots containing the measures such as vehicle count, travel time, rewards and q losses
        The data is over the episodes and for now works only with learning agents
        """
        
        plt.plot(self.veh_count)
        plt.ylabel('vehicle count')
        plt.xlabel('episodes')
        plt.savefig(self.log_path + '/vehCount.png')
        plt.clf()
            
        plt.plot(self.travel_time)
        plt.ylabel('avg travel time')
        plt.xlabel('episodes')
        plt.savefig(self.log_path + '/avgTime.png')
        plt.clf()
            
        plt.plot(self.plot_rewards)
        plt.ylabel('total rewards')
        plt.xlabel('episodes')
        plt.savefig(self.log_path + '/totalRewards.png')
        plt.clf()
        
        plt.plot(self.episode_losses)
        plt.ylabel('q loss')
        plt.xlabel('episodes')
        plt.savefig(self.log_path + '/qLosses.png')
        plt.clf()


    def save_models(self, environ, flag):
        """
        Saves machine learning models (for now just neural networks)
        :param environ: the environment in which the model was run
        """
        if flag:
            torch.save(environ.local_net.state_dict(), self.log_path + '/throughput_q_net.pt')
            torch.save(environ.target_net.state_dict(), self.log_path + '/throughput_target_net.pt')
        else: 
            torch.save(environ.local_net.state_dict(), self.log_path + '/time_q_net.pt')
            torch.save(environ.target_net.state_dict(), self.log_path + '/time_target_net.pt')
