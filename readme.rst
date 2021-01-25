.. highlight:: console

The code uses CityFlow simulation software with several utility functions added. Therefore one needs to download and install my personal fork of CityFlow: https://github.com/mbkorecki/CityFlow.
To run the simulation, change directory to the src folder and call:

.. code-block:: console
				
   python3 traffic_sim.py 
   
This would run the simulation with the default arguments. If you want to specify your own arguments here is what you have at your disposal with example uses:

| --sim_config "../4x4/1.config"
| The path to the cityflow config file for the simulation you want to run

| --num_episodes 1
| The number of episodes you want your learning algorithm to learn for 

| --num_sim_steps 1800
| Number of simulation steps you want the simulation to run for in a single episode

| --agents_type "learning"
| The type of agents you want to run, for now available options are: learning or analytical

| --update_freq 10
| How often the reinforcement learning agent updates its q-network

| --batch_size 64
| The size of the mini-batch used to train the deep-q-network

| --lr 5e-4
| The learning rate

Example call would look like this:

.. code-block:: console
				
   python3 traffic_sim.py --agents_type 'analytical' --sim_config '../4x4/1.config' --num_episodes 1 --num_sim_steps 1800

The docs folder contains documentation build files which can be run with eg. make html.
