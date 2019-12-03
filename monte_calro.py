import os
import sys
import datetime
import json
import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm
import simplekml
from multiprocessing import Pool, Value, Array

from Simulator.rocket_param import Rocket
import Simulator.solver as solver
from make_wind import make_law_wind

########## USER INPUT ###########
rocket_config_file = "rocket_config_10km.json"
wind_base_file = "mc_sample_wind.csv"

mass_std = 0.5  # [kg]
inertia_std = 2.0
Lcg_std = 0.05  # [m]
Lcp_std = 0.05  # [m]
thrust_std = 50  # [N]
Isp_std = 5.0  # [s]
CNa_std = 0.5  # [1/rad]

case_number = 12
#################################

apogee_list = Array("f", range(case_number))
pos_landing_lat_list = Array("f", range(case_number))
pos_landing_lon_list = Array("f", range(case_number))

def __init(apogee_array, lat_array, lon_array):
    global shared_apogee_array, shared_lat_array, shared_lon_array
    shared_apogee_array = apogee_array
    shared_lat_array = lat_array
    shared_lon_array = lon_array


def __get_randam(ave, std):
    return np.random.normal(ave, std)

def __make_wind(index):
    log = np.loadtxt(wind_base_file, delimiter=",", skiprows=1)
    alt_array = log[:, 0]
    vel_array = log[:, 1]
    std_array = log[:, 2]
    dir_array = log[:, 3]

    put_vel_array = np.zeros(len(alt_array))
    put_dir_array = np.zeros(len(alt_array))
    
    for i in range(len(alt_array)):
        put_vel_array[i] = __get_randam(vel_array[i], std_array[i])
        put_dir_array[i] = __get_randam(dir_array[i], 15.0)

    np.savetxt("./winder/wind" + str(index) + ".csv", np.c_[alt_array, put_vel_array, put_dir_array], delimiter=",", header="alt,vel,dir", comments="")

def __make_param():
    rocket_config_json = json.load(open(rocket_config_file))
    engine_config_json = json.load(open(rocket_config_json.get('System').get('Engine Config json')))

    rocket_config_json["Structure"]["Dry Mass [kg]"] = __get_randam(rocket_config_json.get("Structure").get("Dry Mass [kg]"), mass_std)
    rocket_config_json["Structure"]["Dry Length-C.G. [m]"] = __get_randam(rocket_config_json["Structure"]["Dry Length-C.G. [m]"], Lcg_std)
    rocket_config_json["Structure"]["Dry Inertia-Moment Pitch-Axis [kg*m^2]"] = __get_randam(rocket_config_json["Structure"]["Dry Inertia-Moment Pitch-Axis [kg*m^2]"], inertia_std)
    rocket_config_json["Aero"]["Constant Length-C.P. from Nosetip [m]"] = __get_randam(rocket_config_json["Aero"]["Constant Length-C.P. from Nosetip [m]"], Lcp_std)
    rocket_config_json["Aero"]["Constant Normal Coefficient CNa"] = __get_randam(rocket_config_json["Aero"]["Constant Normal Coefficient CNa"], CNa_std)

    engine_config_json["Engine"]["Constant Thrust [N]"] = __get_randam(engine_config_json["Engine"]["Constant Thrust [N]"], thrust_std)
    engine_config_json["Engine"]["Isp [sec]"] = __get_randam(engine_config_json["Engine"]["Isp [sec]"], Isp_std)

    return rocket_config_json, engine_config_json

def __solver(rocket_config_json, engine_config_json):
    model_name = rocket_config_json.get('System').get('Name')

    result_dir = './Result_single_' + model_name
    # if os.path.exists(result_dir):
    #     resultdir_org = result_dir
    #     i = 1
    #     while os.path.exists(result_dir):
    #         result_dir = resultdir_org + '_%02d' % (i)
    #         i = i + 1
    # os.mkdir(result_dir)

    rocket = Rocket(rocket_config_json, engine_config_json, result_dir)
    solver.solve_trajectory(rocket)
    rocket.result.output_min(rocket)
    return rocket

def __runner(itr):
    __make_wind(itr)
    rocket_config_json, engine_config_json = __make_param()
    rocket_config_json["Wind"]["Wind File"] = "./winder/wind" + str(itr) + ".csv"
    rocket = __solver(rocket_config_json, engine_config_json)
    shared_apogee_array[itr] = rocket.result.alt_apogee
    shared_lat_array[itr] = rocket.result.pos_hard_LLH_log[-1, 0]
    shared_lon_array[itr] = rocket.result.pos_hard_LLH_log[-1, 1]


def __plot_kml(lat_array, lon_array, name=''):
    kml = simplekml.Kml()
    for i in range(case_number):
        kml_point = kml.newpoint()
        p = [(lon_array[i], lat_array[i])]
        kml_point.coords = p
        kml_point.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
    kml.save("./landing_plot.kml")


def monte_calro_solver():
    initial_num_list = list(range(case_number))
    with Pool(processes=7, initializer=__init, initargs=(apogee_list, pos_landing_lat_list, pos_landing_lon_list)) as p:
        p.map(func=__runner, iterable=initial_num_list)
    __plot_kml(pos_landing_lat_list, pos_landing_lon_list)

    print(np.array(apogee_list).mean())
    print(np.array(apogee_list).std()) 

    np.savetxt("log.csv", np.c_[pos_landing_lat_list, pos_landing_lon_list], fmt="%.8f", delimiter=",", header="lat,lon", comments="")

    for i in range(case_number):
        os.remove("./winder/wind" + str(i) + ".csv")


if __name__ == '__main__':
    print('Start Time: ', datetime.datetime.now(), 'UTC')
    time_start_calc = datetime.datetime.now()

    monte_calro_solver()

    time_end_calc = datetime.datetime.now()
    print('Calculate Time: ', time_end_calc - time_start_calc)
    print('End Time: ', datetime.datetime.now(), 'UTC')




