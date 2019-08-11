from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.baxter import BaxterEnv

from robosuite.models.objects import CylinderObject, PlateWithHoleObject
from robosuite.models.arenas import EmptyArena
from robosuite.models.robots import Baxter
from robosuite.models import MujocoWorldBase


class BaxterRandomPoints(BaxterEnv):
    """
    This class corresponds to the peg in hole task for the Baxter robot. There's
    a cylinder attached to one gripper and a hole attached to the other one.
    """

    def __init__(
            self,
            reward_shaping=True,
            **kwargs
    ):
        """
        Args:
            cylinder_radius (2-tuple): low and high limits of the (uniformly sampled)
                radius of the cylinder

            cylinder_length (float): length of the cylinder

            use_object_obs (bool): if True, include object information in the observation.

            reward_shaping (bool): if True, use dense rewards

        Inherits the Baxter environment; refer to other parameters described there.
        """


        # reward configuration
        self.reward_shaping = reward_shaping
        self.random_point = np.array([1,1,1])
        self.old_reward=0
        self.counter=0

        super().__init__(gripper_left=None, gripper_right=None, **kwargs)



    def _load_model(self):
        """
        Loads the peg and the hole models.
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # Add arena and robot
        self.model = MujocoWorldBase()
        self.arena = EmptyArena()
        if self.use_indicator_object:
            self.arena.add_pos_indicator()
        self.model.merge(self.arena)
        self.model.merge(self.mujoco_robot)


    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flattened array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()


    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.viewer.viewer.add_marker(pos=self.random_point, label=str(self.counter))

    def reward(self, action):
        """
        Reward function for the task.

        The sparse reward is 0 if the peg is outside the hole, and 1 if it's inside.
        We enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        The dense reward has four components.

            Reaching: in [0, 1], to encourage the arms to get together.
            Perpendicular and parallel distance: in [0,1], for the same purpose.
            Cosine of the angle: in [0, 1], to encourage having the right orientation.
        """
        reward = 0


        x_pos_right=self.sim.data.get_body_xipos("right_hand")
        x_pos_left=self.sim.data.get_body_xipos("left_hand")
        d=np.minimum(np.linalg.norm(x_pos_right-self.random_point),np.linalg.norm(x_pos_left-self.random_point))

        reward=self.old_reward+np.exp(-d)
        del self.viewer.viewer._markers[:]
        self.viewer.viewer.add_marker(pos=self.random_point, label=str(self.counter))
        if d<0.1:
            self.old_reward=reward
            self.random_point = np.array([np.random.uniform(-0.1, 0.7, 1)[0],np.random.uniform(-0.7, 0.7, 1)[0],np.random.uniform(0.5,1, 1)[0]])


            self.counter+=1
            del self.viewer.viewer._markers[:]
            self.viewer.viewer.add_marker(pos=self.random_point, label=str(self.counter))

        # Right location and angle

        return reward

    def _peg_pose_in_hole_frame(self):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.
        """
        # World frame
        peg_pos_in_world = self.sim.data.get_body_xpos("cylinder")
        peg_rot_in_world = self.sim.data.get_body_xmat("cylinder").reshape((3, 3))
        peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

        # World frame
        hole_pos_in_world = self.sim.data.get_body_xpos("hole")
        hole_rot_in_world = self.sim.data.get_body_xmat("hole").reshape((3, 3))
        hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)

        world_pose_in_hole = T.pose_inv(hole_pose_in_world)

        peg_pose_in_hole = T.pose_in_A_to_pose_in_B(
            peg_pose_in_world, world_pose_in_hole
        )
        return peg_pose_in_hole

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()


        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        contact_geoms = (
                self.gripper_right.contact_geoms() + self.gripper_left.contact_geoms()
        )
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                    self.sim.model.geom_id2name(contact.geom1) in contact_geoms
                    or self.sim.model.geom_id2name(contact.geom2) in contact_geoms
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task is successfully completed.
        """


        return self.old_reward>200
