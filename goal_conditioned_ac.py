import tensorflow as tf

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import utils

def set_goal(traj, goal):
  for obs_field in ['observation', 'goal']:
    assert obs_field in traj.observation.keys()
  obs = traj.observation['observation']
  tf.nest.assert_same_structure(obs, goal)
  modified_traj = traj._replace(
      observation={'observation': obs, 'goal': goal})
  return modified_traj

def merge_obs_goal(observations):

  obs = observations['observation']
  goal = observations['goal']

  assert obs.shape == goal.shape
  assert len(obs.shape) == 2
  modified_observations = tf.concat([obs, goal], axis=-1)
  assert obs.shape[0] == modified_observations.shape[0]
  assert modified_observations.shape[1] == obs.shape[1] + goal.shape[1]
  return modified_observations


class GoalConditionedActorNetwork(actor_network.ActorNetwork):
  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               **kwargs):
    modified_tensor_spec = None
    super(GoalConditionedActorNetwork, self).__init__(
        modified_tensor_spec, output_tensor_spec,
        fc_layer_params=(256, 256),
        **kwargs)
    self._input_tensor_spec = input_tensor_spec

  def call(self, observations, step_type=(), network_state=()):
    modified_observations = merge_obs_goal(observations)
    return super(GoalConditionedActorNetwork, self).call(
        modified_observations, step_type=step_type, network_state=network_state)


class GoalConditionedCriticNetwork(critic_network.CriticNetwork):
  def __init__(self,
               input_tensor_spec,
               observation_conv_layer_params=None,
               observation_fc_layer_params=(256,),
               action_fc_layer_params=None,
               joint_fc_layer_params=(256,),
               activation_fn=tf.nn.relu,
               name='CriticNetwork',
               output_dim=None):

    self._output_dim = output_dim
    (_, action_spec) = input_tensor_spec
    modified_obs_spec = None
    modified_tensor_spec = (modified_obs_spec, action_spec)

    super(critic_network.CriticNetwork, self).__init__(
        input_tensor_spec=modified_tensor_spec,
        state_spec=(),
        name=name)
    self._input_tensor_spec = input_tensor_spec

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    self._observation_layers = utils.mlp_layers(
        observation_conv_layer_params,
        observation_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='observation_encoding')

    self._action_layers = utils.mlp_layers(
        None,
        action_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='action_encoding')

    self._joint_layers = utils.mlp_layers(
        None,
        joint_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='joint_mlp')

    self._joint_layers.append(
        tf.keras.layers.Dense(
            self._output_dim if self._output_dim is not None else 1,
            activation=None,
            kernel_initializer=tf.compat.v1.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='value'))

  def call(self, inputs, step_type=(), network_state=()):
    observations, actions = inputs
    modified_observations = merge_obs_goal(observations)
    modified_inputs = (modified_observations, actions)
    output = super(GoalConditionedCriticNetwork, self).call(
        modified_inputs, step_type=step_type, network_state=network_state)
    (predictions, network_state) = output

    if self._output_dim is not None:
      predictions = tf.reshape(predictions, [-1, self._output_dim])

    return predictions, network_state