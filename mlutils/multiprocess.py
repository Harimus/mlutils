import math
import multiprocessing as mp
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

import mlutils.wandb as mlutils_wandb


@cache
def find_grid(num_images, image_width, image_height):
    """Finds the best grid size for displaying num_images images of given width and height.
    Returns the number of rows and columns that minimizes the circumference of the grid.
    Inputs:
        num_images: int, number of images to display
        image_width: int, width of each image
        image_height: int, height of each image
    Returns:
        best_row: int, number of rows in the grid"""
    best_row = best_col = float("inf")
    for cols in range(1, num_images + 1):
        rows = math.ceil(num_images / cols)  # Minimum rows needed
        total_width = cols * image_width
        total_height = rows * image_height
        circ = total_width + total_height
        if circ < (best_row * image_height) + (best_col * image_width):
            best_row = rows
            best_col = cols
    return best_row, best_col


class MultiEnv:
    """A wrapper for multiple environments that allows for parallel execution.
    It handles state history and action length, and provides methods for resetting
    and stepping through the environments.
    Args:
        env_maker: function that creates a new environment instance.
        num_envs: number of environments to create.
        state_history: number of previous states to keep in the state tensor.
        action_length: number of actions to keep in the action tensor.
    """

    def __init__(self, env_maker: Callable, num_envs, state_history=1, action_length=1):
        self.state_history = state_history
        self.action_length = action_length
        self.envs = [env_maker() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.active_envs = [True for _ in range(num_envs)]
        self.latest_state = [None for _ in range(num_envs)]
        self.latest_render = [None for _ in range(num_envs)]

    def close(self):
        """Closes all environments."""
        [env.close() for env in self.envs]

    def render(self):
        """Renders the current state of all active environments.
        Returns a grid image of all active environments(assumes env.render() return image).
        If an environment is not active, it uses the last rendered image.
        Returns:
            grid_image: numpy array of shape (rows * h, cols * w, c) where
            rows and cols are the number of rows and columns in the grid,
            h and w are the height and width of each image, and c is the number of channels.
        """
        images = [
            env.render() if self.active_envs[i] else self.latest_render[i]
            for i, env in enumerate(self.envs)
        ]
        self.latest_render = images

        h, w, c = images[0].shape
        rows, cols = find_grid(len(images), w, h)
        grid_image = np.zeros((rows * h, cols * w, c), dtype=images[0].dtype)

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            grid_image[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = img
        return grid_image

    def concat_states(self, states):
        """States: list of numpy arrays of size [state_dim].
        it concatenate it to the shape [num_envs, state_dim].
        (works the same if staes are [state_dim, state_history])""",
        npstates = np.stack(states, axis=0)
        return torch.from_numpy(npstates).float()

    def reset(self, seed=None):
        """Resets all environments and returns the initial states.
        If seed is provided, it will be used to seed the environments.
        If seed is None, a random seed will be generated.
        Returns:
            states: torch tensor of shape [num_envs, state_dim, state_history].
            Each state is repeated state_history times.
        """
        if seed is None:
            seed = np.random.randint(0, 1e6)
        states = [env.reset(seed=seed + i)[0] for i, env in enumerate(self.envs)]
        states = [state[:, None].repeat(self.state_history, axis=1) for state in states]
        self.active_envs = [True for _ in range(self.num_envs)]
        self.latest_state = states
        self.latest_render = [None for _ in range(self.num_envs)]
        return self.concat_states(states)

    def step(self, action: torch.Tensor, state_history=None):
        """Assumes that actions are torch tensors of form:
        [num_envs, action_size, action_length].
        returns states, rewards, dones, terminate, infos.
        State is of the shape [num_envs, state_size].
        Optionally can take a state history tensor, and add the new state to it.
        In this case, the state returned is [num_envs, state_size, state_history]."""
        states = []
        rewards = [0 for _ in range(self.num_envs)]
        dones = [True for _ in range(self.num_envs)]
        terminates = [True for _ in range(self.num_envs)]
        infos = [None for _ in range(self.num_envs)]
        action = action.detach().cpu().numpy()
        for i, env in enumerate(self.envs):
            if self.active_envs[i]:
                a = action[i]
                a = a[:, 0]  # only using first action
                state, reward, done, terminate, info = env.step(a)
                self.latest_state[i] = state
                states.append(state)
                rewards[i] = reward
                if done or terminate:
                    self.active_envs[i] = False
                dones[i] = done  # Done is True if the episode is done
                terminates[i] = (
                    terminate  # Terminate is True if the environment is done
                )
                infos[i] = info
        states = self.concat_states(self.latest_state)
        if state_history is not None:
            states = torch.cat([state_history[:, :, 1:], states[:, :, None]], dim=-1)
        return states, rewards, dones, terminates, infos


def mp_evaluate(eval_func, env_maker, model_loader, config, queue, wandb_run=None):
    """Function to evaluate a model in a separate process.
    Args:
        eval_func: function to evaluate the model. It should take the model,
        env, config, and other parameters as input. and return total rewards and means, stds.
        env_maker: function to create a new environment instance. Takes config as input and renders if specified.
        model_loader: function to load the model. Takes config and env as input, and returns the model (torch model) that will be run
        config: a DictConfig.
        queue: a multiprocessing queue to get the weight filenames.
        wandb_run: optional Weights & Biases run object to log results.
    """
    device = config.eval.device
    env = env_maker(config, render=config.eval.render)
    model = model_loader(config, env(), device)
    done = False
    myproc = mp.current_process()
    while not done:
        weight_filename = queue.get()
        if weight_filename == "":
            done = True
            print(f"{myproc.name} is shutting down")
            break
        print(f"{myproc.name} will process: {weight_filename}")
        weights = torch.load(weight_filename, map_location=device)
        model.load_state_dict(weights)
        total_rewards, (means, stds) = eval_func(
            model,
            env,
            config,
        )
        print(f"Total rewards: {total_rewards}")
        if wandb_run:
            weight_fname = Path(weight_filename).stem
            weight_number = int(weight_fname.split("_")[-1])
            mlutils_wandb.wandb_log_reward(
                name="reward",
                run=wandb_run,
                reward=np.mean(total_rewards),
                step=weight_number,
            )
            if config.eval.render:
                mlutils_wandb.wandb_log_video(
                    run=wandb_run,
                    name="video",
                    video_path=f"{weight_filename}.mp4",
                )


class MultiProcessTrainEval:
    def __init__(
        self,
        config: DictConfig,
        train_func: Callable,
        eval_func: Callable,
        num_evals: int = 5,
        wandb_run=True,
    ):
        """A class to handle multi-process training and evaluation.
        Args:
            config: a DictConfig object containing the configuration for training and evaluation.
            train_func: a function that takes config, queue, and wandb_run as input and performs training.
            eval_func: a function that takes a model, environment, and config as input and performs evaluation.
            num_evals: number of evaluation processes to spawn.
            wandb_run: optional wandb flag if wandb logging is enabled.
        """
        # See: https://github.com/google-deepmind/mujoco/issues/991
        # torch.multiprocessing.set_start_method("spawn", force=True)
        mp_context = mp.get_context("spawn")
        self.config = config
        self.run = None
        if wandb_run:
            self.run = mlutils_wandb.wandb_from_config(config)
        self.train_func = train_func
        self.queue = mp_context.Queue()
        self.eval_func = eval_func
        self.num_evals = num_evals
        # function can not be class (ref) method when spawning multiple
        self.evaluator_processes = [
            mp_context.Process(
                target=mp_evaluate,
                args=(
                    self.eval_func,
                    self.config,
                    self.queue,
                    self.run,
                ),
            )
            for _ in range(self.num_evals)
        ]
        self.train_process = mp_context.Process(
            target=self.train, args=(self.queue, self.run)
        )

    def start(self):
        self.start_train()
        self.start_eval()
        self.train_process.join()
        for _ in range(self.num_evals):
            self.queue.put("")
        for evaluator_process in self.evaluator_processes:
            evaluator_process.join()
        self.run.finish()

    def start_eval(self):
        for evaluator_process in self.evaluator_processes:
            evaluator_process.start()

    def start_train(self):
        self.train_process.start()

    def train(self, queue: mp.Queue, wandb_run=None):
        self.train_func(self.config, queue, wandb_run)
