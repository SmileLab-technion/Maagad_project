---
layout: page
mathjax: true
---
# Papers Name:

- [Variable Impedance Control](#VIC)



<hr />


<a name='VIC'></a>
##### Variable Impedance Control summarize:

Resent research in RL has focused on "observations to torques". using the torque as an action space has a lot of downsides:
* The torques learned in simulation will only work for this specific robot.
* It’s not possible to change the robot speed/accelaration without learning the parameters again.
* It’s harder to control the robot and the robot transition when the objective and the learned policy are not on the same space.
* Contacts vector is in the direction of the gripper motion(3’rd newton law)

The writer of the paper sugest the end-effector space as the action space. The chllenge working in this space is the transition
between end-effector and the joints representation, this transition is usally done using inverse kinematics. Inverse
kinamatics is aמ optimization process(there is more than one solution for end-effector position) which is time consuming and
as some accuracy disadventage. The writer of the paper solve this problem by learning the controller of the robot, this method 
doesn't need to go throw inverse kinematics. The paper compares diffrent kinds of controllers and compare them to the used torque
action. The final results indicate that impedance controller achive the best results of them all. Impedance controll mean you learn
the position and orientation of the end effector and the kp and kv the spring and damper coef.  



<hr />
