from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import CustomEnv

# instantiate your custom environment
wrapped_env = CustomEnv(1)

# Assuming 'wrapped_env' is your wrapped environment instance
check_env(wrapped_env)