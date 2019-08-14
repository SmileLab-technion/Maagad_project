import numpy as np
import robosuite as suite
import time
from robosuite.controllers import PPO
import sys
from mujoco_py import (MjSim, load_model_from_xml,functions,
                       load_model_from_path, MjSimState,
                       ignore_mujoco_warnings,
                       load_model_from_mjb)
import time
import threading
import math

def quat_to_angles(q):
    angle=np.zeros((3))
    angle[0]=np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))
    angle[1]=np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    angle[2]=np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]))

    return angle

def calc_angle(env):

    quat_cylinder=env.sim.data.get_body_xquat('right_hand')
    quat_hole = env.sim.data.get_body_xquat('left_hand')

    return quat_to_angles(quat_cylinder),quat_to_angles(quat_hole)

def create_stiffness_funct(const_k):
    return const_k*np.identity(3)

def function_Cre():
    # create environment instance

    #env = suite.make("BaxterPegInHole", has_renderer=False, has_offscreen_renderer=False)
    env= suite.make("BaxterRandomPoints", has_renderer=True)


    env.reset()

    target_jacp_cylinder = np.zeros(3 * env.sim.model.nv)
    target_jacr_cylinder= np.zeros(3 * env.sim.model.nv)
    target_jacp_hole = np.zeros(3 * env.sim.model.nv)
    target_jacr_hole= np.zeros(3 * env.sim.model.nv)



    # [K_pos,C_pos,K_ori,C_ori ]
    stiffnes_cons=[2000,100,200,10]
    #A_pos_diag=1e-3
    [K_pos, C_pos, K_ori, C_ori]=[create_stiffness_funct(i) for i in stiffnes_cons]


    pos_desired=env.random_point
    quat_hole= env.random_point
    ori_desired=0

    v_desired=np.array([0,0,0])
    w_desired=np.array([0,0,0])

    F_left_arm=np.zeros((6))
    F_right_arm = np.zeros((6))



    for i in range(10000):
        pos_desired = env.random_point
        action = np.zeros((env.dof))
        H = np.zeros(env.sim.model.nv*env.sim.model.nv)
        functions.mj_fullM(env.sim.model, H, env.sim.data.qM)
        #x, y = math.cos(i), math.sin(i)
        #env.viewer.viewer.add_marker(pos=np.array([x, y, 1]),label=str(i))


        H_use=H.reshape(env.sim.model.nv,env.sim.model.nv)

        pos_hole_now = env.sim.data.get_body_xipos('left_hand')
        v_hole_now = env.sim.data.get_body_xvelp('left_hand')
        w_hole_now= env.sim.data.get_body_xvelr('left_hand')

        pos_cylinder_now = env.sim.data.get_body_xipos("right_hand")
        v_cylinder_now = env.sim.data.get_body_xvelp("right_hand")
        w_cylinder_now= env.sim.data.get_body_xvelr('right_hand')

        [ori_cylinder_now, ori_hole_now] = calc_angle(env)

        F_left_arm[:3] = (np.dot(K_pos, pos_desired - pos_hole_now) + np.dot(C_pos,v_desired-v_hole_now))
        F_left_arm[3:] = 0*(np.dot(K_ori, ori_desired - ori_hole_now) + np.dot(C_ori,w_desired-w_hole_now))

        F_right_arm[:3] = (np.dot(K_pos, pos_desired-pos_cylinder_now) + np.dot(C_pos,v_desired-v_cylinder_now))
        F_right_arm[3:] = 0*(np.dot(K_ori, ori_desired-ori_cylinder_now ) + np.dot(C_ori,w_desired-w_cylinder_now))

        env.sim.data.get_body_jacp('right_hand', jacp=target_jacp_cylinder)
        env.sim.data.get_body_jacr('right_hand', jacr=target_jacr_cylinder)
        env.sim.data.get_body_jacp('left_hand', jacp=target_jacp_hole)
        env.sim.data.get_body_jacr('left_hand', jacr=target_jacr_hole)

        J_L_cylinder = target_jacp_cylinder.reshape((3, env.sim.model.nv))[:,1:8]
        J_A_cylinder = target_jacr_cylinder.reshape((3, env.sim.model.nv))[:,1:8]
        J_L_hole = target_jacp_hole.reshape((3, env.sim.model.nv))[:,8:]
        J_A_hole = target_jacr_hole.reshape((3, env.sim.model.nv))[:,8:]

        #J = np.concatenate((J_L, J_A), axis=0)

        H_L_cylinder = np.dot(np.dot(np.linalg.pinv(J_L_cylinder.T),H_use[1:8,1:8]), np.linalg.pinv(J_L_cylinder))
        H_A_cylinder  =np.dot(np.dot(np.linalg.pinv(J_A_cylinder.T),H_use[1:8,1:8]), np.linalg.pinv(J_A_cylinder))

        H_L_hole = np.dot(np.dot(np.linalg.pinv(J_L_hole.T),H_use[8:,8:]), np.linalg.pinv(J_L_hole))
        H_A_hole  =np.dot(np.dot(np.linalg.pinv(J_A_hole.T),H_use[8:,8:]), np.linalg.pinv(J_A_hole))

        if pos_desired[1]<0:
            action[:7]=env.sim.data.qfrc_bias[1:8]+np.dot(J_L_cylinder.T,np.dot(H_L_cylinder, F_right_arm[:3]))+np.dot(J_A_cylinder.T,np.dot(H_A_cylinder, F_right_arm[3:]))
        #else:
        else:
            action[7:]=env.sim.data.qfrc_bias[8:]+np.dot(J_L_hole.T,np.dot(H_L_hole, F_left_arm[:3]))+np.dot(J_A_hole.T,np.dot(H_A_hole, F_left_arm[3:]))


        #action[:7]=env.sim.data.qfrc_bias[:7]+np.dot(H_use,np.dot(J_L.T,F[:3]))
        obs, reward, done, info = env.step(action)  # take action in the environment
        if done:
            env.reset()
            pos_desired = env.sim.data.get_body_xpos('hole')
            quat_hole = env.sim.data.get_body_xquat('hole')
            ori_desired = quat_to_angles(quat_hole)
        env.render()  # render on display





if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    thread = threading.Thread(target=function_Cre())
    thread.start()