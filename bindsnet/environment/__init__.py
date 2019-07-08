import gym
import torch
import numpy as np

from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

from ..datasets import Dataset, MNIST, CIFAR10, CIFAR100, SpokenMNIST
from ..datasets.preprocess import subsample, gray_scale, binary_image, crop


class Environment(ABC):
    # language=rst
    """
    Abstract environment class.
    """

    @abstractmethod
    def step(self, a: int) -> Tuple[Any, ...]:
        # language=rst
        """
        Abstract method head for ``step()``.

        :param a: Integer action to take in environment.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        # language=rst
        """
        Abstract method header for ``reset()``.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        # language=rst
        """
        Abstract method header for ``render()``.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        # language=rst
        """
        Abstract method header for ``close()``.
        """
        pass

    @abstractmethod
    def preprocess(self) -> None:
        # language=rst
        """
        Abstract method header for ``preprocess()``.
        """
        pass

    @abstractmethod
    def reshape(self) -> None:
        # language=rst
        """
        Abstract method header for ``reshape()``.
        """
        pass


class DatasetEnvironment(Environment):
    # language=rst
    """
    A wrapper around any object from the ``datasets`` module to pass to the ``Pipeline`` object.
    """

    def __init__(self, dataset: Dataset, train: bool = True, time: int = 350, **kwargs):
        # language=rst
        """
        Initializes the environment wrapper around the dataset.

        :param dataset: Object from datasets module.
        :param train: Whether to use train or test dataset.
        :param time: Length of spike train per example.
        :param kwargs: Raw data is multiplied by this value.
        """
        self.dataset = dataset
        self.train = train
        self.time = time

        # Keyword arguments.
        self.intensity = kwargs.get('intensity', 1)
        self.max_prob = kwargs.get('max_prob', 1)

        assert 0 < self.max_prob <= 1, 'Maximum spiking probability must be in (0, 1].'

        self.obs = None

        if train:
            self.data, self.labels = self.dataset.get_train()
            self.label_loader = iter(self.labels)
        else:
            self.data, self.labels = self.dataset.get_test()
            self.label_loader = iter(self.labels)

        self.env = iter(self.data)

    def step(self, a: int = None) -> Tuple[torch.Tensor, int, bool, Dict[str, int]]:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``step()`` function.

        :param a: There is no interaction of the network the dataset.
        :return: Observation, reward (fixed to 0), done (fixed to False), and information dictionary.
        """
        try:
            # Attempt to fetch the next observation.
            self.obs = next(self.env)
        except StopIteration:
            # If out of samples, reload data and label generators.
            self.env = iter(self.data)
            self.label_loader = iter(self.labels)
            self.obs = next(self.env)

        # Preprocess observation.
        self.preprocess()

        # Info dictionary contains label of MNIST digit.
        info = {'label' : next(self.label_loader)}

        return self.obs, 0, False, info

    def reset(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``reset()`` function.
        """
        # Reload data and label generators.
        self.env = iter(self.data)
        self.label_loader = iter(self.labels)

    def render(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``render()`` function.
        """
        pass

    def close(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``close()`` function.
        """
        pass

    def preprocess(self) -> None:
        # language=rst
        """
        Preprocessing step for a state specific to dataset objects.
        """
        self.obs = self.obs.view(-1)
        self.obs *= self.intensity

    def reshape(self) -> torch.Tensor:
        # language=rst
        """
        Get reshaped observation for plotting purposes.

        :return: Reshaped observation to plot in ``plt.imshow()`` call.
        """
        if type(self.dataset) == MNIST:
            return self.obs.view(28, 28)
        elif type(self.dataset) in [CIFAR10, CIFAR100]:
            temp = self.obs.view(32, 32, 3).cpu().numpy() / self.intensity
            return temp / temp.max()
        elif type(self.dataset) in SpokenMNIST:
            return self.obs.view(-1, 40)


class GymEnvironment(Environment):
    # language=rst
    """
    A wrapper around the OpenAI ``gym`` environments.
    """

    def __init__(self, name: str, **kwargs) -> None:
        # language=rst
        """
        Initializes the environment wrapper.

        :param name: The name of an OpenAI :code:`gym` environment.

        Keyword arguments:

        :param max_prob: Maximum spiking probability.
        :param clip_rewards: Whether or not to use :code:`np.sign` of rewards.
        """
        self.name = name
        self.env = gym.make(name)
        self.action_space = self.env.action_space

        # Keyword arguments.
        self.max_prob = kwargs.get('max_prob', 1)
        self.clip_rewards = kwargs.get('clip_rewards', True)

        self.obs = None
        self.reward = None

        assert 0 < self.max_prob <= 1, 'Maximum spiking probability must be in (0, 1].'

    def step(self, a: int) -> Tuple[torch.Tensor, float, bool, Dict[Any, Any]]:
        # language=rst
        """
        Wrapper around the OpenAI Gym environment :code:`step()` function.

        :param a: Action to take in the environment.
        :return: Observation, reward, done flag, and information dictionary.
        """
        # Call gym's environment step function.
        self.obs, self.reward, self.done, info = self.env.step(a)

        if self.clip_rewards:
            self.reward = np.sign(self.reward)

        self.preprocess()

        # Return converted observations and other information.
        return self.obs, self.reward, self.done, info

    def reset(self) -> torch.Tensor:
        # language=rst
        """
        Wrapper around the OpenAI Gym environment :code:`reset()` function.

        :return: Observation from the environment.
        """
        # Call gym's environment reset function.
        self.obs = self.env.reset()
        self.preprocess()

        return self.obs

    def render(self) -> None:
        # language=rst
        """
        Wrapper around the OpenAI Gym environment :code:`render()` function.
        """
        self.env.render()

    def close(self) -> None:
        # language=rst
        """
        Wrapper around the OpenAI Gym environment :code:`close()` function.
        """
        self.env.close()

    def preprocess(self) -> None:
        # language=rst
        """
        Pre-processing step for an observation from a Gym environment.
        """
        if self.name == 'SpaceInvaders-v0':
            self.obs = subsample(gray_scale(self.obs), 84, 110)
            self.obs = self.obs[26:104, :]
            self.obs = binary_image(self.obs)
        elif self.name == 'BreakoutDeterministic-v4':
            self.obs = subsample(gray_scale(crop(self.obs, 34, 194, 0, 160)), 80, 80)
            self.obs = binary_image(self.obs)
        else: # Default pre-processing step
            self.obs = subsample(gray_scale(self.obs), 84, 110)
            self.obs = binary_image(self.obs)

        self.obs = torch.from_numpy(self.obs).float()

    def reshape(self) -> torch.Tensor:
        # language=rst
        """
        Reshape observation for plotting purposes.

        :return: Reshaped observation to plot in ``plt.imshow()`` call.
        """
        return self.obs
