import numpy as np
import mujoco_py as mj
from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer
from mujoco_py import (MjSim, load_model_from_xml,functions,MjViewer,
                       load_model_from_path, MjSimState,
                       ignore_mujoco_warnings,
                       load_model_from_mjb)


from matplotlib import pyplot as plt
import math

import time

def quat_to_angles(q):
    angle=np.zeros((3))
    angle[0]=np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))
    angle[1]=np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    angle[2]=np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]))

    return angle
def rotation_matrix_to_angles(mat):
    theta = -np.arcsin(mat[2, 0])
    cos_theta =np.arccos(theta)
    psi = np.arctan2(mat[2, 1] / cos_theta, mat[2, 2] / cos_theta)
    phi = np.arctan2(mat[1, 0] / cos_theta, mat[0, 0] / cos_theta)

    return [phi*180/np.pi, theta*180/np.pi,psi*180/np.pi]

if __name__ == '__main__':

    xml = """
    <mujoco model="example">
        <compiler coordinate="global"/>
        <default>
            <geom rgba=".8 .6 .4 1"/>
        </default>
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".6 .8 1" 
                     width="256" height="256"/>
        </asset>
        <worldbody>
            <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
            <geom name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="10 10 10" type="plane"/>
            <body>
                <site name="world" size="0.1" pos="0 0 0" />
                
                <geom name="first_pole" type="capsule" fromto="0 0 0  0 0 0.5" size="0.04"/>
                <joint name='a' type="hinge" pos="0 0 0" axis="0 0 1" />
                <body name="second_pole">
                    <inertial pos="0 0 0" mass="0.00000001" diaginertia="1e-008 1e-008 1e-008" />
                    <geom type="capsule" fromto="0 0 0.5  0.5 0 0.5" size="0.04" name="second_pole"/>
                    <joint name='b' type="hinge" pos="0 0 0.5" axis="0 1 0"/>      
                    <body name='third_pole'>
                        <inertial pos="0 0 0" mass="0.00000001" diaginertia="1e-008 1e-008 1e-008" />
                        <geom type="capsule" fromto="0.5 0 0.5  1 0 0.5" size="0.04" name="third_pole"/>
                        <joint name='c' type="hinge" pos="0.5 0 0.5" axis="0 1 0"/>    
                        <site name="target" size="0.1" pos="1 0 0.5" />
                        <body name="mass">
                            <inertial pos="1 0 0.5" mass="1e-2" diaginertia="1e-008 1e-008 1e-008" />
                            <geom type="sphere" pos="1 0 0.5" size="0.2" name="mass"/>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor joint="a"/>
            <motor joint="b"/>
            <motor joint="c"/>
        </actuator>	
        
    
    </mujoco>
    """

    model = load_model_from_xml(xml)

    sim = MjSim(model)
    #viewer = MujocoPyRenderer(sim)
    viewer=MjViewer(sim)

    sim.reset()
        # After reset jacobians are all zeros
    sim.forward()
    target_jacp = np.zeros(3 * sim.model.nv)
    target_jacr= np.zeros(3 * sim.model.nv)



    F=np.array([0,0,-9.81*1e-2,0,0,0]).T

    #np.testing.assert_allclose(target_jacp, np.zeros(3 * sim.model.nv))
        # After first forward, jacobians are real
    #sim.forward()
    K_diag=200
    C_diag=10

    K_diag_rot=200
    C_diag_rot=10


    A_diag=1e-3

    K=[np.identity(3)*K_diag,np.identity(3)*K_diag_rot]
    C=[np.identity(3)*C_diag,np.identity(3)*C_diag_rot]
    A=np.identity(3)*A_diag



    #K_diag=0.3
    #C_diag=0.05



    x_intial=sim.data.site_xpos[1]
    print(x_intial)
    x_desired=np.random.uniform(-np.sqrt(0.5),np.sqrt(0.5),3)
    x_desired[2]=np.abs(x_desired[2])

    v_intial=sim.data.site_xvelp[1]
    v_desired=np.array([0,0,0])

    a_desired=np.array([0,0,0])
    a_intial=np.array([0,0,0])


    dt=sim.model.opt.timestep
    #sim.data.get_site_jacp('target', jacp=target_jacp)
        # Should be unchanged after steps (zero action)
    graph=[]
    desired_angle=[0,0,np.pi/2]
    point_acom=1
    for i in range(100000):

        x, y = math.cos(i), math.sin(i)

        viewer.add_marker(pos=x_desired,label=str(point_acom))

        quat = sim.data.get_body_xquat('mass')
        angle = quat_to_angles(quat)
        angle_velocity=sim.data.get_body_xvelp('mass')

        F[:3]=np.dot(K[0],x_desired-x_intial)+np.dot(C[0],v_desired-v_intial)+np.dot(A,a_desired-a_intial)
        F[3:]=np.dot(K[1],desired_angle-angle)+np.dot(C[1],-angle_velocity)

        H = np.zeros(sim.model.nv* sim.model.nv)
        functions.mj_fullM(sim.model, H, sim.data.qM)

        sim.data.get_site_jacp('target', jacp=target_jacp)
        sim.data.get_site_jacr('target', jacr=target_jacr)
        J_L = target_jacp.reshape((3, sim.model.nv))
        J_A = target_jacr.reshape((3, sim.model.nv))
        J = np.concatenate((J_L, J_A), axis=0)

        H_A = np.dot(np.linalg.pinv(J_A.T), np.dot(H.reshape(sim.model.nv, sim.model.nv), np.linalg.pinv(J_A)))
        H_L =np.dot(np.linalg.pinv(J_L.T),np.dot(H.reshape(sim.model.nv, sim.model.nv), np.linalg.pinv(J_L)))
        H_all=np.dot(np.linalg.pinv(J.T),np.dot(H.reshape(sim.model.nv, sim.model.nv), np.linalg.pinv(J)))
        #F_a=np.dot(A,0.3-sim.data.qacc)
        action = sim.data.qfrc_bias+np.dot(J_L.T, np.dot(H_L, F[:3]))
        #action = np.dot(J_L.T, np.dot(H_L, F[:3]))+sim.data.qfrc_bias
        #action = sim.data.qfrc_bias+np.dot(H.reshape(3,3),np.dot(J_L.T,F[:3]))
        #print(action)
        #action =  np.dot(J.T, F)
        sim.data.ctrl[:] = action
        sim.step()
        sim.forward()
        #print(np.max(action))
        #print(sim.data.qacc)
        viewer.render()
        x_intial = sim.data.site_xpos[1]
        a_intial=(v_intial-sim.data.site_xvelp[1])/dt
        v_intial = sim.data.site_xvelp[1]
        normal=np.linalg.norm(x_desired-x_intial)
        print(normal)
        #print(normal)

        # if normal<0.1:
        #     print("in")
        #     if desired_angle[2]==np.pi/2:
        #         desired_angle = np.array([0, 0, 0])
        #     else:
        #         desired_angle = np.array([0, 0, np.pi / 2])

        if normal<0.3:
            point_acom+=1
            x_desired=np.random.uniform(-np.sqrt(0.5),np.sqrt(0.5),3)
            x_desired[2] = np.abs(x_desired[2])
            del viewer._markers[:]

        graph.append(np.abs(x_intial-x_desired))
     #   sim.forward()


    print("the desired is {} and the intial is{}".format(x_desired,x_intial))
    plt.plot(graph)
    plt.show()