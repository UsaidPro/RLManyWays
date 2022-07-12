import torch

"""
Implementation of the Policy Iteration Algorithm detailed on pg. 97 on the gridworld problem in PyTorch
TODO: Possible Improvements
 - Convert state for loop to iterating over a tensor of indices
"""

grid = torch.zeros(size=(5,5))
policy = torch.randint(low=0, high=4, size=(5, 5))  # Randomly set actions
actions = torch.tensor(
    [
        [-1, 0],  # Up 
        [0, 1],   # Right
        [1, 0],   # Down
        [0, -1]   # Left
        ], dtype=torch.int64)
possible_states = []
for i in range(5):
    for j in range(5):
        if (i == 0 and j == 1) or (i == 0 and j == 3):
            continue
        possible_states.append([i, j])
states = torch.tensor(possible_states, dtype=torch.int64)

def reward(new_state):
    # Defines discounted expected value reward for a state
    min_pos = torch.min(new_state)
    max_pos = torch.max(new_state)
    if min_pos < 0 or max_pos > 4:
        clamped_state = torch.clamp(new_state, 0, 4)
        return -1 + 0.9 * grid[tuple(clamped_state)]
    else:
        return 0.9 * grid[tuple(new_state)]

def evaluate_policy(policy):
    # Evaluates every state using policy to get corresponding values
    maxdiff = torch.tensor(10)
    while maxdiff > 0.000001:
        # 3 components to the gridworld: At state A, At state B, Elsewhere
        # At state A. All output actions result in 10 reward and being put on A'
        prev = grid[0, 1]
        grid[0, 1] = 10 + 0.9 * grid[4, 1]
        maxdiff = torch.abs(grid[0, 1] - prev) 
        # At state B. Same reward as A just that reward is now 5
        prev = grid[0, 3]
        grid[0, 3] = 5 + 0.9 * grid[2, 3]
        maxdiff = torch.max(torch.abs(grid[0, 3] - prev), maxdiff)
        # All other states do value iteration
        for state in states:
            state_tuple = tuple(state)
            resultant_state = state + actions[policy[state_tuple]]
            prev = grid[state_tuple]
            grid[state_tuple] = reward(resultant_state)
            maxdiff = torch.max(torch.abs(grid[state_tuple] - prev), maxdiff)
            # grid[i][j] = policy[i, j, 0] * reward(i - 1, j) + policy[i, j, 1] * reward(i + 1, j) + policy[i, j, 2] * reward(i, j - 1) + policy[i, j, 3] * reward(i, j + 1)

def improve_policy(policy):
    # Sample actions using policy probabilities
    # For each state, do the argmax action looking at resultant state values.
    # Over time it will converge to the optimal values/policy
    done = True
    for i in range(5):
        for j in range(5):
            possible_indices = actions + torch.tensor([i, j])
            max_idx = 0
            max_value = -9999
            for actions_idx, grid_index in enumerate(possible_indices):
                # Handle policy going to invalid states
                clamped_index = grid_index.clamp(0, 4)
                cell_value = grid[tuple(clamped_index)]
                if max_value <= cell_value:
                    max_idx = actions_idx
                    max_value = cell_value
            if max_idx != policy[i, j]:
                done = False
            policy[i, j] = max_idx
    return done

evaluate_policy(policy)
for i in range(500):
    print(f'ITERATION {i}')
    if(improve_policy(policy=policy)):
        break
    evaluate_policy(policy)
    print("VALUES:")
    for i in range(5):
        print(grid[i])
    print("POLICY:")
    for i in range(5):
        print(policy[i])
