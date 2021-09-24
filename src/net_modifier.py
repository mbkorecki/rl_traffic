import cityflow

import argparse
import os
import random
import json
import queue
import numpy as np
import scipy.stats as stats

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default='../scenarios/4x4mount/',  type=str, help="the relative directory of the sim files")
    parser.add_argument("--sim_config", default='../scenarios/4x4mount/1.config',  type=str, help="the relative path to the simulation config file")
    parser.add_argument("--roadnet", default='../scenarios/4x4mount/roadnet_m_m.json',  type=str, help="the relative path to the simulation roadnet file")
    parser.add_argument("--flow", default='../scenarios/4x4mount/anon_4_4_700_0.3.json',  type=str, help="the relative path to the flow file")

    return parser.parse_args()



args = parse_args()

num_points = [0, 1, 2, 3, 4, 5]
div = {1:2, 2:4, 3:5, 4:6, 5:7}


def disrupt_veh_speed(args):
    with open(args.flow, "r") as flow_file:
        data = json.load(flow_file)

        mu, sigma = 15, 5
        lower, upper = 9, 20
        
        for vehicle in data:
            vehicle['vehicle']['maxSpeed'] =  stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=15, scale=5)
            # print(vehicle['vehicle']['maxSpeed'])

    with open(args.dir + "flow_m.json", "w") as flow_file:
        json.dump(data, flow_file)

        
    with open(args.roadnet, "r") as roadnet_file:
        data = json.load(roadnet_file)
        for road in data['roads']:
            road['maxSpeed'] = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=15, scale=5)
            for lane in road['lanes']:
                lane['maxSpeed'] = road['maxSpeed']

            
    with open(args.dir + "roadnet_m_m_m.json", "w") as roadnet_file:
        json.dump(data, roadnet_file)
        
def disrupt_road_topology(args):
    with open(args.roadnet, "r") as roadnet_file:
        data = json.load(roadnet_file)
        intersections = {}
        points = {}

        for road in data['roads']:
            flag = False
            point1 = road['points'][0]
            point2 = road['points'][1]

            diff_x = point1['x'] - point2['x']
            diff_y = point1['y'] - point2['y']


            num_point = random.sample(num_points, 1)[0]

            if random.random() > 0.5: change = 50
            else: change = -50


            if (road['endIntersection'], road['startIntersection']) in points.keys():
                num_point = points.get((road['endIntersection'], road['startIntersection']))
            points.update({(road['startIntersection'], road['endIntersection']) : num_point})

            if (road['endIntersection'], road['startIntersection']) in intersections.keys():
                change = intersections.get((road['endIntersection'], road['startIntersection']))
                flag = True
            intersections.update({(road['startIntersection'], road['endIntersection']) : change})

            new_points = []

            if not (diff_x == 0 and diff_y == 0):
                for i in range(num_point):
                    if diff_x == 0:
                        new_point = {'x' : point1['x'] + change, 'y' : min(point1['y'], point2['y']) + (i+1) * abs(point1['y'] - point2['y']) / div[num_point]}
                    if diff_y == 0:
                        new_point = {'x' : min(point1['x'], point2['x']) + (i+1) * abs(point1['x'] - point2['x']) / div[num_point], 'y' : point1['y'] + change}
                    new_points.append(new_point)
                    change *= -1

                if new_points:
                    print(point1, point2)
                    print(new_points)
                    if flag:
                        new_points.reverse()
                    road['points'][1:1] = new_points

    with open(args.dir + "roadnet_m_m.json", "w") as roadnet_file:
        json.dump(data, roadnet_file)


def draw_tikzpicture(args):
    scale = 1/100
    latex_string = "\\begin{tikzpicture}\n"
    
    with open(args.roadnet, "r") as roadnet_file:
        data = json.load(roadnet_file)
        
        for road in data['roads']:
            for i in range(len(road['points']) - 1):
                latex_string += "\draw " + "(" + str(road['points'][i]['x'] * scale) + "," + str(road['points'][i]['y'] * scale) + ")" + " -- " + "(" + str(road['points'][i+1]['x'] * scale) + "," + str(road['points'][i+1]['y'] * scale) + ")" ";"


        for intersection in data['intersections']:
            if not intersection['virtual']:
                latex_string += "\\filldraw [blue]" + "(" + str(intersection['point']['x'] * scale) + ", " + str(intersection['point']['y'] * scale) + ")" + "circle (8pt);"

    latex_string += "\n\end{tikzpicture}"
    with open("../tikzpicture.txt", "w") as save_file:
        save_file.write(latex_string)



def get_road_lengths(args):
    with open(args.roadnet, "r") as roadnet_file:
        data = json.load(roadnet_file)

    road_lengths = []
    for road in data['roads']:
        road_length = 0
        for i in range(len(road['points']) - 1):
            road_length += np.sqrt((road['points'][i+1]['x'] - road['points'][i]['x'])**2 + (road['points'][i+1]['y'] - road['points'][i]['y'])**2)
        road_lengths.append(road_length)

    print(np.mean(road_lengths), np.std(road_lengths))


def get_flow_rates(args):
    with open(args.flow, "r") as flow_file:
        data = json.load(flow_file)

    length = 3600
    vehs_time_array = np.zeros(length)
    last_time = -1

    starting_points = {}
    
    for elem in data:
        if elem['startTime'] >= length:
            continue
        if elem['route'][0] not in starting_points.keys():
            starting_points.update({elem['route'][0] :  np.zeros(length)})
        for i in range(elem['startTime'], elem['endTime']+1):
            vehs_time_array[i] += 1
            starting_points[elem['route'][0]][i] += 1

    arr_rate = np.zeros(length)
    for i in range(length):
        arr_rate[i] = np.sum(vehs_time_array[0:i+1]) / (i+1)

    # print(np.mean(arr_rate), np.var(arr_rate))
    # print(starting_points.keys(), len(starting_points.keys()))

    arr_points_sum = []
    for val in starting_points.values():
        print(np.sum(val), np.mean(val), np.var(val))
        arr_points_sum.append(np.sum(val))

    print(np.mean(arr_points_sum), np.var(arr_points_sum))
    print(np.sum(vehs_time_array), np.mean(vehs_time_array), np.var(vehs_time_array))

get_flow_rates(args)
# get_road_lengths(args) 
# draw_tikzpicture(args)            
# disrupt_veh_speed(args)
# disrupt_road_topology(args)
