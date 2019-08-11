import numpy as np
import robosuite as suite
import time
from robosuite.controllers import PPO


# create environment instance
env = suite.make("BaxterPegInHole", has_renderer=False, has_offscreen_renderer=False)
render_env= suite.make("BaxterPegInHole", has_renderer=True)

PPO_control=PPO(Env=env,Render_env=render_env,n_iter=400,paths=20,horizon=50,Load_nn=False,exp_name="BaxterPegInHolePPO_end_effector_1",critic=False)
PPO_control.initiate_ppo_controler()
PPO_control.train()
#PPO_control.save_tf_model()
#PPO_control.finish_sess()
#PPO_control.load_tf_model()
#PPO_control.train()
#PPO_control.finish_sess()
#PPO_control.render(render_env,250)
#PPO_control.render(render_env,100)
