import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import Callable, Optional

from ..network import Network
from ..encoding import bernoulli
from ..network.nodes import Input, AbstractInput
from ..environment import Environment
from ..network.monitors import Monitor
from ..analysis.plotting import plot_spikes, plot_voltages

__all__ = [
    'Pipeline', 'action'
]

plt.ion()


class Pipeline:
    # language=rst
    """
    Abstracts the interaction between network, environment (or dataset), input encoding, and environment feedback
    action.
    """

    def __init__(self, network: Network, environment: Environment, encoding: Callable = bernoulli,
                 action_function: Optional[Callable] = None, enable_history: Optional[bool] = False,
                 **kwargs):
        # language=rst
        """
        Initializes the pipeline.

        :param network: Arbitrary network object.
        :param environment: Arbitrary environment.
        :param encoding: Function to encode observations into spike trains.
        :param action_function: Function to convert network outputs into environment inputs.
        :param enable_history: Enable history functionality.

        Keyword arguments:

        :param int plot_interval: Interval to update plots.
        :param str save_dir: Directory to save network object to.
        :param int print_interval: Interval to print text output.
        :param int time: Time input is presented for to the network.
        :param int history: Number of observations to keep track of.
        :param int delta: Step size to save observations in history.
        :param bool render_interval: Interval to render the environment.
        :param int save_interval: How often to save the network to disk.
        :param str output: String name of the layer from which to take output from.
        :param float plot_length: Relative time length of the plotted record data. Relative to parameter time.
        :param str plot_type: Type of plotting ('color' or 'line').
        :param int reward_delay: How many iterations to delay delivery of reward.
        """
        self.network = network
        self.env = environment
        self.encoding = encoding
        self.action_function = action_function
        self.enable_history = enable_history

        self.episode = 0
        self.iteration = 0
        self.history_index = 1
        self.s_ims, self.s_axes = None, None
        self.v_ims, self.v_axes = None, None
        self.obs_im, self.obs_ax = None, None
        self.reward_im, self.reward_ax = None, None
        self.accumulated_reward = 0
        self.reward_list = []

        # Setting kwargs.
        self.time = kwargs.get('time', 1)
        self.delta = kwargs.get('delta', 1)
        self.output = kwargs.get('output', None)
        self.save_dir = kwargs.get('save_dir', 'network.pt')
        self.plot_interval = kwargs.get('plot_interval', None)
        self.save_interval = kwargs.get('save_interval', None)
        self.print_interval = kwargs.get('print_interval', None)
        self.history_length = kwargs.get('history_length', None)
        self.render_interval = kwargs.get('render_interval', None)
        self.plot_length = kwargs.get('plot_length', 1.0)
        self.plot_type = kwargs.get('plot_type', 'color')
        self.reward_delay = kwargs.get('reward_delay', None)

        self.dt = network.dt
        self.timestep = int(self.time / self.dt)

        if self.history_length is not None and self.delta is not None:
            self.history = {i: torch.Tensor() for i in range(1, self.history_length * self.delta + 1, self.delta)}
        else:
            self.history = {}

        if self.plot_interval is not None:
            for l in self.network.layers:
                self.network.add_monitor(Monitor(self.network.layers[l], 's', int(self.plot_length * self.plot_interval * self.timestep)),
                                         name=f'{l}_spikes')
                if 'v' in self.network.layers[l].__dict__:
                    self.network.add_monitor(Monitor(self.network.layers[l], 'v', int(self.plot_length * self.plot_interval * self.timestep)),
                                             name=f'{l}_voltages')

            self.spike_record = {l: torch.Tensor().byte() for l in self.network.layers}
            self.set_spike_data()
            self.plot_data()

        if self.reward_delay is not None:
            assert self.reward_delay > 0
            self.rewards = torch.zeros(self.reward_delay)

        # Set up for multiple layers of input layers.
        self.encoded = {
            name: torch.Tensor() for name, layer in network.layers.items() if isinstance(layer, AbstractInput)
        }

        self.obs = None
        self.reward = None
        self.done = None

        self.voltage_record = None

        self.first = True
        self.clock = time.time()

    def step(self, **kwargs) -> None:
        # language=rst
        """
        Run an iteration of the pipeline.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if neurons should be clamped to
                                              spiking. The ``Tensor``s have shape ``[n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks if neurons should be clamped
                                                to not spiking. The ``Tensor``s should have shape ``[n_neurons]``.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to boolean masks determining which
                                                     weights to clamp to zero.
        :param float max_prob: Maximum probability of firing for ``bernoulli`` spike train encoder.
        """
        if self.print_interval is not None and self.iteration % self.print_interval == 0:
            print(f'Iteration: {self.iteration} (Time: {time.time() - self.clock:.4f})')
            self.clock = time.time()

        if self.save_interval is not None and self.iteration % self.save_interval == 0:
            print(f'Saving network to {self.save_dir}')
            self.network.save(self.save_dir)

        # Render game.
        if self.render_interval is not None and self.iteration % self.render_interval == 0:
            self.env.render()

        # Choose action based on output neuron spiking.
        if self.action_function is not None:
            a = self.action_function(self, output=self.output)
        else:
            a = None

        # Run a step of the environment.
        self.obs, reward, self.done, info = self.env.step(a)

        if self.reward_delay is not None:
            self.rewards = torch.tensor([reward, *self.rewards[1:]]).float()
            self.reward = self.rewards[-1]
        else:
            self.reward = reward

        # Store frame of history and encode the inputs.
        if self.enable_history and len(self.history) > 0:
            self.update_history()
            self.update_index()

        # Encode the observation using given encoding function.
        for inpt in self.encoded:
            self.encoded[inpt] = self.encoding(self.obs, time=self.time, dt=self.network.dt, **kwargs)

        # Run the network on the spike train-encoded inputs.
        self.network.run(inpts=self.encoded, time=self.time, reward=self.reward, **kwargs)

        # Plot relevant data.
        if self.plot_interval is not None and self.iteration % self.plot_interval == 0:
            self.plot_data()

            if self.iteration > len(self.history) * self.delta:
                self.plot_obs()

        self.iteration += 1

        if self.done:
            self.iteration = 0
            self.episode += 1
            self.reward_list.append(self.accumulated_reward)
            self.accumulated_reward = 0
            self.plot_reward()

    def plot_obs(self) -> None:
        # language=rst
        """
        Plot the processed observation after difference against history
        """
        if self.obs_im is None and self.obs_ax is None:
            fig, self.obs_ax = plt.subplots()
            self.obs_ax.set_title('Observation')
            self.obs_ax.set_xticks(())
            self.obs_ax.set_yticks(())
            self.obs_im = self.obs_ax.imshow(self.env.reshape(), cmap='gray')
        else:
            self.obs_im.set_data(self.env.reshape())

    def plot_reward(self) -> None:
        """
        Plot the change of accumulated reward for each episodes
        """
        if self.reward_im is None and self.reward_ax is None:
            fig, self.reward_ax = plt.subplots()
            self.reward_ax.set_title('Reward')
            self.reward_plot, = self.reward_ax.plot(self.reward_list)
        else:
            reward_array = np.array(self.reward_list)
            y_min = reward_array.min()
            y_max = reward_array.max()
            self.reward_ax.set_xlim(left=0, right=self.episode)
            self.reward_ax.set_ylim(bottom=y_min, top=y_max)
            self.reward_plot.set_data(range(self.episode), self.reward_list)

    def plot_data(self) -> None:
        # language=rst
        """
        Plot desired variables.
        """
        # Set latest data
        self.set_spike_data()
        self.set_voltage_data()

        # Initialize plots
        if self.s_ims is None and self.s_axes is None and self.v_ims is None and self.v_axes is None:
            self.s_ims, self.s_axes = plot_spikes(self.spike_record)
            self.v_ims, self.v_axes = plot_voltages(self.voltage_record,
                    plot_type=self.plot_type, threshold=self.threshold_value)
        else:
            # Update the plots dynamically
            self.s_ims, self.s_axes = plot_spikes(self.spike_record, ims=self.s_ims, axes=self.s_axes)
            self.v_ims, self.v_axes = plot_voltages(self.voltage_record, ims=self.v_ims,
                    axes=self.v_axes, plot_type=self.plot_type, threshold=self.threshold_value)

        plt.pause(1e-8)
        plt.show()

    def set_spike_data(self) -> None:
        # language=rst
        """
        Get the spike data from all layers in the pipeline's network.
        """
        self.spike_record = {l: self.network.monitors[f'{l}_spikes'].get('s') for l in self.network.layers}

    def set_voltage_data(self) -> None:
        # language=rst
        """
        Get the voltage data and threshold value from all applicable layers in the pipeline's network.
        """
        self.voltage_record = {}
        self.threshold_value = {}
        for l in self.network.layers:
            if 'v' in self.network.layers[l].__dict__:
                self.voltage_record[l] = self.network.monitors[f'{l}_voltages'].get('v')
            if 'thresh' in self.network.layers[l].__dict__:
                self.threshold_value[l] = self.network.layers[l].thresh

    def update_history(self) -> None:
        # language=rst
        """
        Updates the observations inside history by performing subtraction from  most recent observation and the sum of
        previous observations. If there are not enough observations to take a difference from, simply store the
        observation without any differencing.
        """
        # Recording initial observations
        if self.iteration < len(self.history) * self.delta:
            # Store observation based on delta value
            if self.iteration % self.delta == 0:
                self.history[self.history_index] = self.obs
        else:
            # Take difference between stored frames and current frame
            temp = torch.clamp(self.obs - sum(self.history.values()), 0, 1)

            # Store observation based on delta value.
            if self.iteration % self.delta == 0:
                self.history[self.history_index] = self.obs

            assert (len(self.history) == self.history_length), 'History size is out of bounds'
            self.obs = temp

    def update_index(self) -> None:
        # language=rst
        """
        Updates the index to keep track of history. For example: history = 4, delta = 3 will produce self.history = {1,
        4, 7, 10} and self.history_index will be updated according to self.delta and will wrap around the history
        dictionary.
        """
        if self.iteration % self.delta == 0:
            if self.history_index != max(self.history.keys()):
                self.history_index += self.delta
            else:
                # Wrap around the history.
                self.history_index = (self.history_index % max(self.history.keys())) + 1

    def reset_(self) -> None:
        # language=rst
        """
        Reset the pipeline.
        """
        self.env.reset()
        self.network.reset_()
        self.iteration = 0
        self.history = {i: torch.Tensor() for i in self.history}
