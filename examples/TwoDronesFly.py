import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.OLControl import OLControl
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_AGGREGATE = True
DEFAULT_DURATION_SEC = 3
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONE, 
        num_drones=DEFAULT_NUM_DRONES,
        gui=DEFAULT_GUI, 
        physics=DEFAULT_PHYSICS,
        record_video=DEFAULT_RECORD_VIDEO, 
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, 
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, 
        aggregate=DEFAULT_AGGREGATE, 
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        plot=DEFAULT_PLOT,
        colab=DEFAULT_COLAB
    ):
    #### Initialize the simulation #############################
    arm_in_x = 0.0379/np.sqrt(2)
    arm_in_x_projection = arm_in_x * np.cos(0.5)

    # INIT_XYZS = np.array([[-0.458, 0, 1],[0.458, 0, 2]])
    INIT_XYZS = np.array([[0, 0, 1] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(num_drones)])
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz) if aggregate else 1 # if aggregate=ture --> AGGR_PHY_STEPS=int(...)

    #### Initialize the target rpy ###########################
    TARGET_RPY = np.array([[0, 0.5, 0] for i in range(num_drones)])

    #### Create the environment ###################
    env = CtrlAviary(drone_model=drone,
                    num_drones=num_drones,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    physics=physics,
                    neighbourhood_radius=10,
                    freq=simulation_freq_hz,
                    aggregate_phy_steps=AGGR_PHY_STEPS,
                    gui=gui,
                    record=record_video,
                    obstacles=False
                    )

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=num_drones,
                    duration_sec=duration_sec,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    ctrl = [OLControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz)) # np.floor turns into integer
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(num_drones)}
    START = time.time()
    for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, _, _, _ = env.step(action)

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:
            #### Compute control for the current way point #############
            for j in range(num_drones):
                if i <= 200:
                    action[str(j)] = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                    state=obs[str(j)]["state"],
                                                                    target_pos=INIT_XYZS[j, :],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )
                elif i>200 and i <400:
                    action[str(j)] = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(j)]["state"],
                                                                target_pos=INIT_XYZS[j, :],
                                                                target_rpy=np.array([0, 0.5, 0])
                                                                )
                elif i>400 and i<600:
                    action[str(j)] = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(j)]["state"],
                                                                target_pos=INIT_XYZS[j, :],
                                                                target_rpy=np.array([0, -0.5, 0])
                                                                )
                else:
                    action[str(j)] = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(j)]["state"],
                                                                target_pos=INIT_XYZS[j, :],
                                                                target_rpy=np.array([0, 0, 0])
                                                                )

            
        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state=obs[str(j)]["state"],
                       control=np.hstack([np.zeros(6), TARGET_RPY[j], np.zeros(3)])
                       )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if gui:
            #sync(i, START, env.TIMESTEP) # adjust timestep to slow down the speed of the sim
            sync(i, START, 0.0075)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("dw") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Downwash example script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--aggregate',          default=DEFAULT_AGGREGATE,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))