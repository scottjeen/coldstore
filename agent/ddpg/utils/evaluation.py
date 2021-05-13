def evaluate_agent(env, agent):
    rb_control = env.data.copy()
    agent_control = env.data.copy()

    # create graph with dummy variables before we can load model
    n_steps = 0
    observation, index = env.reset(start=False)
    while n_steps <= agent.batch_size:
        action = agent.choose_action(observation, evaluate=False)
        observation_, index_, reward, done = env.step(observation, action, index, evaluate=False)
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        index = index_
        n_steps += 1
    agent.learn()
    agent.load_model()

    # begin agent evaluation
    observation, index = env.reset(start=True)
    env.step_cntr = 0
    done = False
    while not done:
        action = agent.choose_action(observation, evaluate=True)
        observation_, index_, reward, done = env.step(observation, action, index, evaluate=True) # need to update step function to take evaluate arg
        agent_control.iloc[index, env.action_idx[0].tolist()] = env.action_kw
        agent_control.iloc[index, env.target_idx] = env.temps

        observation = observation_
        index = index_

        print('step:', env.step_cntr)

    return rb_control, agent_control

# def temp_test()
