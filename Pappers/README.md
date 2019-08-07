---
layout: default
mathjax: true
---
# Papers Name:

- [Variable Impedance Control](#VIC)
- [Accurace_Dilemma_Impedance_Control](#ADIC)



<hr />
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
kinamatics is a optimization process(there is more than one solution for end-effector position) which is time consuming and
as some accuracy disadventage. The writer of the paper solve this problem by learning the controller of the robot, this method 
doesn't need to go throw inverse kinematics. The paper compares diffrent kinds of controllers and compare them to the used torque
action. The final results indicate that impedance controller achive the best results of them all. Impedance controll mean you learn
the position and orientation of the end effector and the kp and kv the spring and damper coef.  

<hr />
<a name='ADIC'></a>

##### Accurace Dilemma Impedance Control summarize:

controlling the force and controlling the position can act as Contradictory forces. for example, if we trained our model to preform drill a 2 mm hole in a metal shit while in reality the shit is made of wood then the force control by himself will create much deeper hole while the position will use much less force combining this two together we have to use hybrid control with parameter s to state which condition is more important and when.  
Another and better approach to this problem is to find a function connecting between force and it’s kinematics or in its scientific name impedance. In this approach we can the write contact equation as a function of the kinematics (between the surface and the manipulator) and state that when equilibrium is achieved then there is static relation between force and position.


$$ \[(1)\,H(q)\ddot{q}+c(q,\dot{q})\dot{q}+G(q)=\tau -{{J}^{T}}{{F}_{int}}\] $$
 

Insteed of looking on F external which we don't have information on we can look on it's complemantry $ F_{int}=-F_{ext}$.
We can look on our tool as spring and damper connected to a mass.

$$ (2) F_{int}=K(x_d-x)+B(\dot{x_d}-\dot{x})+M\ddot{x_d} \to \ddot{x_d}=M^{-1}(F_{int}-K(x_d-x)-B(\dot{x_d}-\dot{x})) $$

xd in this equation can be treated as “virtual” position and can understood as the position if the contact surface resistance would have been 0.



#### Finding $\tau$ :

$$ (3) H(q)\ddot{q}+c(q,\dot{q})\dot{q}+G(q)+J^{T}F_{int}=\tau $$

$$ \dot{X}=J\dot{q},\ddot{X}=\dot{J}\dot{q}+J\ddot{q}\to \ddot{q}={{J}^{-1}}(\ddot{X}-\dot{J}\dot{q}),\dot{q}={{J}^{-1}}\dot{X} $$

$$ (4)H(q){{J}^{-1}}(\ddot{X}-\dot{J}\dot{q})+c(q,\dot{q}){{J}^{-1}}\dot{X}+G(q)+{{J}^{T}}{{F}_{int}}=\tau $$

#### Let's define:

$\scriptsize (J^T)^{-1}H(q)J^{-1}=H* $ and $ \scriptsize (J^T)^{-1}(H(q)J^{-1}\dot{J}\dot{q}+c(q,\dot{q})J^{-1}\dot{X}+G(q))=h*$
and we obtain the folowing equation:

$$ (5) \tau-F_{int} = H*\ddot{x}+h*  $$

If we plug (2) into equation (5) we get the folowing:

$$(6) H{M}^{-1}(F_{int}-K(x_d-x)-B(\dot{x_d}-\dot{x}))-h*+{F}_{int}=\tau $$


##### Impedance controll reduced to Inverse dynamics+PD when $\scriptsize F_{int}=0$:

Let's plug (5) into (1) while setting $ \scriptsize {F}_{int} $ to 0:

$$\mathbf  H{M}^{-1}(-K(x_d-x)-B(\dot{x_d}-\dot{x}))-h*=H*\ddot{x}+h* \to  H{M}^{-1}(\ddot{x}+K(x_d-x)+B(\dot{x_d})=0 $$ 
