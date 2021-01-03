import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

class Logger:
    
    def __init__(self, args):

        self.args = args
        
        self.veh_count = []
        self.travel_time = []
        self.losses = []
        self.plot_rewards = []
        self.episode_losses = []

        self.reward = 0
        
        self.log_path = "results" + args.sim_config.split('/')[0] + '-' + str(args.num_episodes) + '-' + str(args.update_freq)
        old_path = self.log_path
        i = 1

        while os.path.exists(self.log_path):
            self.log_path = old_path + "(" + str(i) + ")"
            i += 1

        os.mkdir(self.log_path)


    def log_measures(self, environ):

        self.reward = 0
        for agent in environ.agents:
            self.reward += (agent.total_rewards / (self.args.num_sim_steps / environ.action_freq))
            agent.total_rewards = 0
        
        self.plot_rewards.append(self.reward)
        self.veh_count.append(environ.eng.get_finished_vehicle_count())
        self.travel_time.append(environ.eng.get_average_travel_time())
        self.episode_losses.append(np.mean(self.losses))
        

    def save_log_file(self, environ):
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
                       "\nmean travel time: " + str(np.mean(self.travel_time[self.args.num_episodes-10:])) + " with sd: " + str(np.std(self.travel_time[self.args.num_episodes-10:]
                        )))
        log_file.write("\n")
        log_file.write("\n")

        for agent in environ.agents:
            log_file.write(agent.ID + "\n")
            for i in range(-1, environ.n_actions):
                log_file.write("phase " + str(i) + " duration: " + str(agent.past_phases.count(i)) + "\n")
            log_file.write("\n")

            for i in range(-1, environ.n_actions):
                log_file.write("phase " + str(i) + " switch: " + str(len(agent.total_duration[i+1])) + "\n")
            log_file.write("\n")

            log_file.write("avg max wait time: " + str(np.mean([x for x in agent.max_wait_time if x != 0])) + "\n")
            for i in range(12):
                log_file.write("movement " + str(i) + " max wait time: " + str(agent.max_wait_time[i]) + "\n")
            log_file.write("\n")
    
        log_file.write("\n")
        
        log_file.close()



    def save_phase_plots(self, environ):
        for agent in environ.agents:
            plt.plot(agent.past_phases, '|', linewidth=25)
            figure = plt.gcf()
            figure.set_size_inches(20,10)
            plt.xticks(np.arange(0, self.args.num_sim_steps+1, step=10))
            plt.ylabel('phase')
            plt.xlabel('time')
            plt.grid()
            ax = plt.gcf().get_axes()[0]
            ax.spines['left'].set_position(('data', 0))
            ax.spines['bottom'].set_position(('data', 0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.savefig(self.log_path + '/phase' + str(agent.ID) + '.png', bbox_inches='tight')
            plt.clf()


    def save_measures_plots(self):
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


    def save_models(self, environ):
        torch.save(environ.local_net.state_dict(), self.log_path + '/policy_net.pt')
        torch.save(environ.target_net.state_dict(), self.log_path + '/target_net.pt')

