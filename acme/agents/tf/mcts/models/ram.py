# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An environment dynamics model based on RAM states."""

from typing import Optional, Tuple

from acme import specs
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts.models import base
from acme.tf import utils as tf2_utils

from bsuite.baselines.utils import replay
import dm_env
import numpy as np
from scipy import special
import sonnet as snt
import tensorflow as tf

class RepresentationMLP(snt.Module):
	def __init__(
		self,
		input_size,
		layer_sizes,
		output_size,
		output_activation=tf.identity,
		activation=tf.nn.elu,
	):
		super(RepresentationMLP, self).__init__(name="Muzero_representation_mlp")
		self.sizes = [input_size] + layer_sizes + [output_size]
		self.layers = []
		for i in range(len(sizes)-1):
			act = activation if i < len(sizes) - 2 else output_activation
			layers += [snt.Linear(sizes[i], sizes[i+1]), act()]
		self.MLP = snt.Sequential(layers)

	def __call__(self, x):
		x = self.MLP(x)
		return x

class MLPRepresentationModel(snt.Module):
	"""This uses MLPs to encode the initial observation into a hidden state."""

	def __init__(
			self,
			environment_spec: specs.EnvironmentSpec,
			hidden_sizes: Tuple[int, ...],
			representation_layers,
			dynamics_layers,
			reward_layers,
			support_size,
			stacked_observations
	):
		super(MLPRepresentationModel, self).__init__(name='MuZero_representation_model')

		# Get num actions/observation shape.
		self._num_actions = environment_spec.actions.num_values
		self._input_shape = environment_spec.observations.shape
		self._flat_shape = int(np.prod(self._input_shape))

		# Prediction networks.
		self._representation_network = RepresentationMLP(
			self._input_shape[0]
			* self._input_shape[1]
			* self._input_shape[2] 
			* (stacked_observations + 1)
			+ stacked_observations * self._input_shape[1] * self._input_shape[2],
			representation_layers,
			hidden_sizes
		)

	def __call__(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
		# there will almost certainly be a dimension error lol
		encoded_state = snt.Flatten()(state)
		encoded_state = self._representation_network(encoded_state)

		# Scale encoded state between [0, 1] (See paper appendix Training)
        min_encoded_state = next_state.min(1, keepdim=True)[0]
        max_encoded_state = next_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state

		return encoded_state_normalized


class DynamicsMLP(snt.Module):
	def __init__(
		self,
		input_size,
		layer_sizes,
		output_size,
		output_activation=tf.identity,
		activation=tf.nn.elu,
	):
		super(DynamicsMLP, self).__init__(name="MuZero_dynamics_mlp")
		self.sizes = [input_size] + layer_sizes + [output_size]
		self.layers = []
		for i in range(len(sizes)-1):
			act = activation if i < len(sizes) - 2 else output_activation
			layers += [snt.Linear(sizes[i], sizes[i+1]), act()]
		self.MLP = snt.Sequential(layers)

	def __call__(self, x):
		x = self.MLP(x)
		return x


class MLPDynamicsModel(snt.Module):
	"""This uses MLPs to model (s, a) -> (r, d, s'). NOTE: s is not an image, it's a hidden state"""

	def __init__(
			self,
			environment_spec: specs.EnvironmentSpec,
			hidden_sizes: Tuple[int, ...],
			dynamics_layers,
			reward_layers,
			support_size,
	):
		super(MLPTransitionModel, self).__init__(name='MuZero_dynamics_model')

		# Get num actions/observation shape.
		self._num_actions = environment_spec.actions.num_values
		self._input_shape = environment_spec.observations.shape
		self._flat_shape = int(np.prod(self._input_shape))

		# Prediction networks.
		self._dynamics_encoded_state_network = DynamicsMLP(
			hidden_sizes + self._num_actions,
			dynamics_layers, # needs to be defined!
			hidden_sizes
		)

		self._dynamics_reward_network = DynamicsMLP(
			hidden_sizes,
			reward_layers, # needs to be defined!
			self.full_support_size
		)

		self._discount_network = snt.Sequential([
				snt.nets.MLP(hidden_sizes + (1,)),
				lambda d: tf.squeeze(d, axis=-1),
		])

	def __call__(self, state: tf.Tensor, action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

		# Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
		embedded_state = snt.Flatten()(state)
		embedded_action = tf.one_hot(action, depth=self._num_actions)

		embedding = tf.concat([embedded_state, embedded_action], axis=-1)

		# Predict the next state, reward, and termination.
		next_state = self._dynamics_encoded_state_network(embedding)
		reward = self._dynamics_reward_network(embedding)
		discount_logits = self._discount_network(embedding)

		# Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

		return next_encoded_state_normalized, reward, discount_logits

class MuZeroDynamicsModel(base.Model):
	"""MuZero's Dynamics network, based on Werner Duvaud's implementation."""

	_checkpoint: types.Observation
	_state: types.Observation

	def __init__(
			self,
			environment_spec: specs.EnvironmentSpec,
			replay_capacity: int,
			batch_size: int,
			hidden_sizes: Tuple[int, ...],
			learning_rate: float = 1e-3,
			terminal_tol: float = 1e-3,
			dynamics_layers,
			reward_layers,
			support_size,
	):
		self._obs_spec = environment_spec.observations
		self._action_spec = environment_spec.actions
		# Hyperparameters.
		self._batch_size = batch_size
		self._terminal_tol = terminal_tol

		# Modelling
		self._replay = replay.Replay(replay_capacity)
		self._transition_model = MLPDynamicsModel(
			environment_spec,
			hidden_sizes,
			dynamics_layers,
			reward_layers,
			support_size,
		)
		self._representation_model = MLPRepresentationModel(
			environment_spec,
			hidden_sizes,
			dynamics_layers,
			reward_layers,
			support_size
		)
		self._optimizer = snt.optimizers.Adam(learning_rate)
		self._forward = tf.function(self._transition_model)
		tf2_utils.create_variables(
				self._transition_model, [self._obs_spec, self._action_spec])
		
		self._variables = self._transition_model.trainable_variables \
			+ self._representation_model.trainable_variables

		# Model state.
		self._needs_reset = True

	@tf.function
	def _step(
			self,
			o_t: tf.Tensor,
			a_t: tf.Tensor,
			r_t: tf.Tensor,
			d_t: tf.Tensor,
			o_tp1: tf.Tensor,
	) -> tf.Tensor:

		with tf.GradientTape() as tape:
			next_state, reward, discount = self._transition_model(o_t, a_t)

			state_loss = tf.square(next_state - o_tp1)
			reward_loss = tf.square(reward - r_t)
			discount_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_t, discount)

			loss = sum([
					tf.reduce_mean(state_loss),
					tf.reduce_mean(reward_loss),
					tf.reduce_mean(discount_loss),
			])

			reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)

		gradients = tape.gradient(loss, self._variables)
		self._optimizer.apply(gradients, self._variables)

		return loss

	def step(self, action: types.Action):
		# Reset if required.
		if self._needs_reset:
			raise ValueError('Model must be reset with an initial timestep.')

		# Step the model.
		state, action = tf2_utils.add_batch_dim([self._state, action])
		new_state, reward, discount_logits = [
				x.numpy().squeeze(axis=0) for x in self._forward(state, action)
		]
		discount = special.softmax(discount_logits)

		# Save the resulting state for the next step.
		self._state = new_state

		# We threshold discount on a given tolerance.
		if discount < self._terminal_tol:
			self._needs_reset = True
			return dm_env.termination(reward=reward, observation=self._state.copy())
		return dm_env.transition(reward=reward, observation=self._state.copy())

	def reset(self, initial_state: Optional[types.Observation] = None):
		if initial_state is None:
			raise ValueError('Model must be reset with an initial state.')
		# We reset to an initial state that we are explicitly given.
		# This allows us to handle environments with stochastic resets (e.g. Catch).
		
		# will this work or fail the type checking? that'll need more work
		self._state = self._representation_model(initial_state)
		self._needs_reset = False
		return dm_env.restart(self._state)

	def update(
			self,
			timestep: dm_env.TimeStep,
			action: types.Action,
			next_timestep: dm_env.TimeStep,
	) -> dm_env.TimeStep:
		# Add the true transition to replay.
		transition = [
				timestep.observation,
				action,
				next_timestep.reward,
				next_timestep.discount,
				next_timestep.observation,
		]
		self._replay.add(transition)

		# Step the model to generate a synthetic transition.
		ts = self.step(action)

		# Copy the *true* state on update.
		self._state = next_timestep.observation.copy()

		if ts.last() or next_timestep.last():
			# Model believes that a termination has happened.
			# This will result in a crash during planning if the true environment
			# didn't terminate here as well. So, we indicate that we need a reset.
			self._needs_reset = True

		# Sample from replay and do SGD.
		if self._replay.size >= self._batch_size:
			batch = self._replay.sample(self._batch_size)
			self._step(*batch)

		return ts

	def save_checkpoint(self):
		if self._needs_reset:
			raise ValueError('Cannot save checkpoint: model must be reset first.')
		self._checkpoint = self._state.copy()

	def load_checkpoint(self):
		self._needs_reset = False
		self._state = self._checkpoint.copy()

	def action_spec(self):
		return self._action_spec

	def observation_spec(self):
		return self._obs_spec

	@property
	def needs_reset(self) -> bool:
		return self._needs_reset

def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = tf.nn.softmax(logits, axis=1)
    support = tf.constant([x for x in range(-support_size, support_size + 1)], dtype=tf.float32)
    support = tf.broadcast_to(support, probabilities.shape)
    x = tf.math.reduce_sum(support * probabilities, axis=1, keepdims=True)
    
    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = tf.math.sign(x) * (
        ((tf.math.sqrt(1 + 4 * 0.001 * (tf.math.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = tf.clip_by_value(x, -support_size, support_size)
    floor = tf.math.floor(x)
    prob = x - floor
    logits = tf.zeros(x.shape[0], x.shape[1], 2 * support_size + 1)
    logits = tf.scatter_nd(
        tf.expand_dims(floor + support_size, -1), tf.expand_dims(1 - prob, -1), shape= # shape required, account for dims=2
    )
    indexes = floor + support_size + 1
    prob = tf.where(2 * support_size < indexes, prob, 0.0)
    indexes = tf.where(2 * support_size < indexes, indexes, 0.0)
    logits.tf.scatter_nd(
    	tf.expand_dims(indexes.long(), -1), tf.expand_dims(prob, -1) shape= # shape required, account for dims=2
    )
    return logits
