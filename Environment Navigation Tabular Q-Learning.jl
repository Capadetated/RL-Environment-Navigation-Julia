using Random

# Initialize parameters
gamma = 0.05  # Discount factor. Adjusted to 0.05 to observe effects.
alpha = 0.05  # Learning rate. Adjusted to 0.05 to observe effects.

# Define the states. 10th state added
location_to_state = Dict(
    "L1" => 1,
    "L2" => 2,
    "L3" => 3,
    "L4" => 4,
    "L5" => 5,
    "L6" => 6,
    "L7" => 7,
    "L8" => 8,
    "L9" => 9,
    "L10" => 10
)

# Define the actions. 10th action added
actions = collect(1:10)

# Define the rewards. A 10th column and row added. L10 is only accessible via L9
rewards = [
    0 1 0 0 0 0 0 0 0 0;
    1 0 1 0 1 0 0 0 0 0;
    0 1 0 0 0 1 0 0 0 0;
    0 0 0 0 0 0 1 0 0 0;
    0 1 0 0 0 0 0 1 0 0;
    0 0 1 0 0 0 0 0 0 0;
    0 0 0 1 0 0 0 1 0 0;
    0 0 0 0 1 0 1 0 1 0;
    0 0 0 0 0 0 0 1 0 1;
    0 0 0 0 0 0 0 0 1 0
]

# Maps indices to locations
state_to_location = Dict(value => key for (key, value) in location_to_state)

function get_optimal_route(start_location::String, end_location::String)
    # Copy the rewards matrix to a new Matrix
    rewards_new = copy(rewards)
    # Get the ending state corresponding to the ending location as given
    ending_state = location_to_state[end_location]
    # With the above information automatically set the priority of the given ending state to the highest one
    rewards_new[ending_state, ending_state] = 999

    # -----------Q-Learning algorithm-----------

    # Initializing Q-Values. Array expanded to 10x10
    Q = zeros(10, 10)

    # Q-Learning process
    for i in 1:1000
        # Pick up a state randomly
        current_state = rand(1:10)
        # For traversing through the neighbor locations in the maze
        playable_actions = Int[]
        # Iterate through the new rewards matrix and get the actions > 0
        for j in 1:9
            if rewards_new[current_state, j] > 0
                push!(playable_actions, j)
            end
        end
        # Pick an action randomly from the list of playable actions leading us to the next state
        next_state = rand(playable_actions)
        # Compute the temporal difference
        # The action here exactly refers to going to the next state
        TD = rewards_new[current_state, next_state] + gamma * Q[next_state, argmax(Q[next_state, :])] - Q[current_state, next_state]
        # Update the Q-Value using the Bellman equation
        Q[current_state, next_state] += alpha * TD
    end

    # Initialize the optimal route with the starting location
    route = [start_location]
    # We don't know about the next location yet, so initialize with the value of the starting location
    next_location = start_location
    # We don't know about the exact number of iterations needed to reach the final location hence while loop will be a good choice for iterating
    steps = 0
    while next_location != end_location
        # Fetch the starting state
        starting_state = location_to_state[start_location]
        # Fetch the highest Q-value pertaining to the starting state
        next_state = argmax(Q[starting_state, :])
        # We got the index of the next state. But we need the corresponding letter.
        next_location = state_to_location[next_state]
        push!(route, next_location)
        # Update the starting location for the next iteration
        start_location = next_location
        # Update steps
        steps += 1
    end
    # Rounding off the Q-values
    Q = round.(Q)
    # For question 9, alternate title L1/L4
    println("1,000 Iterations; L10 to L4")
    println(Q)
    return route, steps
end

# For question 9, this is alternated from L10 to L1, and L10 to L4
println(get_optimal_route("L10", "L1"))
