import tensorflow as tf
from tensorflow.keras import layers  # type: ignore
from mcpi.minecraft import Minecraft
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import load_model  # type: ignore
origin=7,24,-4
def get_position():
    player_pos = mc.player.getTilePos()  # Get the player's position
    # Save the position relative to the spawn point (8.5, 64, 229)
    x = player_pos.x
    y = player_pos.y
    z = player_pos.z
    return x, y, z



def calculate_distance(x, y, z,x_origin=7,y_origin=24,z_origin=-4):
    return abs(x_origin + (x * -1))
    return math.sqrt((x_origin - x)**2  + (z_origin - z)**2)


def step_action(action):
    reward = 0
    done = False
    x0, y0, z0 = get_position()
    distance0 = calculate_distance(x0, y0, z0)
    
    # Define new positions based on the action
    if action == 0:  # Move forward
        new_pos = (x0 + 1, y0, z0)
    elif action == 1:  # Move left
        new_pos = (x0, y0, z0 - 1)
    elif action == 2:  # Move backward
        new_pos = (x0 - 1, y0, z0)
    elif action == 3:  # Move right
        new_pos = (x0, y0, z0 + 1)
    

    x, y, z = new_pos[0],new_pos[1],new_pos[2]
    distance1 = calculate_distance(x, y, z)

    # Get blocks around the player
    blocks = list(mc.getBlocks(x - 1, y - 1, z + 1, x + 1, y +1, z - 1))
    

    if blocks[4] == 0:
        done = True
        return blocks,reward,done
    elif blocks[4] != 0:
        if distance0 > distance1:
            reward += 1
        elif distance0 == distance1:
            reward += 0.5
        else:
            reward -= 1
        # Execute the action in Minecraft
    mc.player.setPos(new_pos[0], new_pos[1], new_pos[2])
    
    
    return blocks, reward, done

def reset():
    x = -4
    y = 24
    z = -3
    print("reset")

    mc.player.setPos(x, y, z)
    distance0=calculate_distance(x,y,z)
    
    # Get the 9 blocks below the player
    blocks = list(mc.getBlocks(x - 1, y - 1, z + 1, x + 1, y +1, z - 1))
    
    # Position 4: block where the player is
    # Position 7: block in front of the player below
    
    return blocks

def create_model(input_size, action_size):
    model = tf.keras.Sequential([
        layers.InputLayer(shape=(input_size,)),
        layers.Dense(158),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(0.2),
        layers.Dense(158),
        layers.ELU(),
        layers.Dense(action_size)  # Output without activation
    ])
    


    return model

mc = Minecraft.create()
mc.postToChat(f"Program Started")

# Create the model
model = create_model(27, 4)

# Initialize variables
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
epsilon = 1.0  # Initial exploration probability
epsilon_decay = 0.995  # Decay factor
epsilon_min = 0.1  # Minimum epsilon value
losses = []

for episode in range(1000):  # Number of episodes
    state= reset()

   
    state = np.array([state])
    total_reward = 0

    for t in range(200):  # Limit steps per episode
        q_values = model(state)
        if np.random.rand() < epsilon:
            action = np.random.randint(4)  # Random action
        else:
            action = np.argmax(q_values.numpy())
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        # Take action and observe the new state and reward
        next_state, reward, done = step_action(action)
        total_reward += reward
        
        # Update the model (using TD-Target for Q-Learning)
        target = reward + 0.33 * np.max(model(np.expand_dims(next_state, axis=0)).numpy())
        target_q_values = q_values.numpy()
        target_q_values[0, action] = target

        with tf.GradientTape() as tape:
            q_values_pred = model(state)
            loss = loss_fn(target_q_values, q_values_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(loss.numpy())  # Save the loss
        state = np.array([next_state])

        if done:
            break

    print(f"Episode: {episode}, Total Reward: {total_reward}, Loss: {loss.numpy()}")

# Plot the loss
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Evolution During Training')
plt.show()
model.save('minecraft_agent_3.h5')
