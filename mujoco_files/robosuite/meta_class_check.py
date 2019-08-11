
REGISTERED_ENVS = {}

def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def make(env_name, *args, **kwargs):
    """Try to get the equivalent functionality of gym.make in a sloppy way."""
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)

class EnvMeta(type):
    """Metaclass for registering environments"""
#__new__ is for creating the object while __init__ is for initilizeing. new always starts before init

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["MujocoEnv", "SawyerEnv", "BaxterEnv"]
        return cls


class MujocoEnv(metaclass=EnvMeta):
    """Initializes a Mujoco Environment."""
    def __init__(self):
        self.shdfsh=3
