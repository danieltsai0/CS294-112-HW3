    def compute_error(obs, obs_tp1, rew, act, done):
        q_values = q_func(obs, num_actions, scope="q_func", reuse=True)
        # greedy q_network action 
        q_values_t = q_func(obs_tp1, num_actions, scope="q_func_t", reuse=True)
        q_value_t = tf.reduce_max(q_values_t)
        # check if terminal state
        if done == 1:
            y = rew   
        else:
            y = tf.add(rew,tf.multiply(gamma, q_value_t))

        return np.square(y - q_values[act])


    

    # # compute total error
    # for j in range(batch_size):
    #     obs, obs_tp1, rew, act, done = obs_t_float[j,:], obs_tp1_float[j,:], rew_t_ph[j], act_t_ph[j], done_mask_ph[j]
    #     q_values = q_func(obs, num_actions, scope="q_func", reuse=True)
    #     # greedy q_network action 
    #     q_values_t = q_func(obs_tp1, num_actions, scope="q_func_t", reuse=True)
    #     q_value_t = tf.reduce_max(q_values_t)
    #     # check if terminal state
    #     if done == 1:
    #         y = rew   
    #     else:
    #         y = tf.add(rew,tf.multiply(gamma, q_value_t))

    #     total_error += np.square(y - q_values[act])
    #     # total_error += compute_error(obs, obs_tp1, rew, act, done)