# Chapter 5, Example 5.3 Page 123
# Implementing Monte Carlo ES on Blackjack
import torch

"""
THIS DOES NOT WORK CORRECTLY RIGHT NOW. NEED TO FIGURE OUT WHY
"""

# Card values. Face cards = 10, Ace = 1
cards = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])

q_values = torch.zeros(size=(10,10,2,2))  # 1 = A...10 dealer showing. 2 = 12...21 player sum, 3 = whether or not usable ace, 4 = action
policy = torch.randint(0, 2, size=(10,10,2), dtype=torch.int64)  # Policy is argmax of q-values so not really necessary but having for vis

# Instead of keeping list of returns for each state-action pair, instead going to recompute average every step
total_return = torch.zeros(size=(10,10,2,2))
times_visited = torch.ones(size=(10,10,2,2), dtype=torch.int64)
visited = torch.ones(size=(10,10,2,2), dtype=torch.int8)

for i in range(100000):
    # Choose state-action pair such that all pairs have nonzero prob
    current_dealer = torch.randint(1, 11, (1,), dtype=torch.int64)
    current_sum = torch.randint(12, 22, (1,), dtype=torch.int64)
    usable_ace = torch.randint(0, 2, (1,), dtype=torch.int64)

    # Generate episode starting from the pair using policy
    episode = []  # Episode list of state-action values. Going to be [(dealer sum, player sum, usable ace), policy]
    while current_sum <= 21:
        episode.append((current_dealer - 1, current_sum - 12, usable_ace, policy[current_dealer - 1, current_sum - 12, usable_ace]))
        if policy[current_dealer - 1, current_sum - 12, usable_ace] == 1:  # Policy is to hit
            new_card = cards[torch.randint(0, 13, (1,), dtype=torch.long)]  # Why does indices have to be long
            current_sum += new_card
            if current_sum > 21 and usable_ace == 1:
                current_sum -= 10
                usable_ace = 0
        else:
            break
    reward = 0
    if current_sum > 21:  # Went bust
        reward = -1
    else:
        # Dealer's turn
        while current_dealer < 17:  # Must hit under 17
            current_dealer += cards[torch.randint(0, 13, (1,), dtype=torch.long)]
        if current_dealer > 21 or current_dealer < current_sum:  # Goes bust or less dealer total, player wins
            reward = 1
    
    # For each pair in episode update pair's value using subsequent return after first occurrence
    for idx, pair in enumerate(episode):
        total_return[pair] += visited[pair] * reward  # Since discount = 0 and reward on non-gameend = 0, then return = end reward
        times_visited[pair] += 1
        visited[pair] -= visited[pair]
        # NOTE: THIS IS NOT FIRST-OCCURRENCE SINCE REVISITING S-A PAIR REUPDATES IT
        # NOW IT IS FIRST OCCURRENCE. SEE HOW THIS CHANGES IT
    q_values = total_return / times_visited
    visited = torch.ones(size=(10,10,2,2), dtype=torch.int8)
    # Update policy doing argmax of q-values
    policy = torch.argmax(q_values, dim=-1)

print(policy[:, :, 0])
print(policy[:, :, 1])