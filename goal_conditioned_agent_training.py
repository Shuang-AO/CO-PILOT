import random
import time
import tqdm
import numpy as np
import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

def train_eval(
        tf_agent,
        tf_env,
        eval_tf_env,
        num_iterations=200000,
        initial_collect_steps=1000,
        batch_size=64,
        num_eval_episodes=100,
        eval_interval=10000,
        log_interval=1000,
        random_seed=0):

    tf.compat.v1.logging.info('random_seed = %d' % random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.compat.v1.set_random_seed(random_seed)

    max_episode_steps = tf_env.pyenv.envs[0]._duration
    global_step = tf.compat.v1.train.get_or_create_global_step()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size)

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=1)

    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    initial_collect_driver.run()

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    for _ in tqdm.tnrange(num_iterations):
        start_time = time.time()
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state,
        )

        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)
        time_acc += time.time() - start_time

        if global_step.numpy() % log_interval == 0:
            tf.compat.v1.logging.info('step = %d, loss = %f', global_step.numpy(),
                                      train_loss.loss)
            steps_per_sec = log_interval / time_acc
            tf.compat.v1.logging.info('%.3f steps/sec', steps_per_sec)
            time_acc = 0

        if global_step.numpy() % eval_interval == 0:
            start = time.time()
            tf.compat.v1.logging.info('step = %d' % global_step.numpy())
            for dist in [2, 5, 10]:
                tf.compat.v1.logging.info('\t dist = %d' % dist)
                eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                    prob_constraint=1.0, min_dist=dist - 1, max_dist=dist + 1)

                results = metric_utils.eager_compute(
                    eval_metrics,
                    eval_tf_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_prefix='Metrics',
                )
                for (key, value) in results.items():
                    tf.compat.v1.logging.info('\t\t %s = %.2f', key, value.numpy())
                pred_dist = []
                for _ in range(num_eval_episodes):
                    ts = eval_tf_env.reset()
                    dist_to_goal = agent._get_dist_to_goal(ts)[0]
                    pred_dist.append(dist_to_goal.numpy())
                tf.compat.v1.logging.info('\t\t predicted_dist = %.1f (%.1f)' %
                                          (np.mean(pred_dist), np.std(pred_dist)))
            tf.compat.v1.logging.info('\t eval_time = %.2f' % (time.time() - start))

    return train_loss, global_step