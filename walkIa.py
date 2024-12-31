from time import sleep
import tensorflow as tf
from tensorflow.keras import layers  # type: ignore
from mcpi.minecraft import Minecraft
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.models import load_model  # type: ignore
import random
from collections import deque

"""
Creates the neural network model and combines it with the Q-Learning algorithm.

Model Input:
The model takes as input the blocks around the player in layers: 
first the blocks below with a distance of one block, then the middle layer, and finally the upper blocks around the character.

Model Output:
Returns a prediction of where the character should move:
0: forward
1: left
2: backward
3: right
"""
origin = 7, 24, -4
mc = Minecraft.create()
mc.postToChat("Program Started")

def get_position():
    player_pos = mc.player.getTilePos()  # Get the player's position
    # Save the position relative to the spawn point (8.5, 64, 229)
    x = player_pos.x
    y = player_pos.y
    z = player_pos.z
    return x, y, z

print(get_position())

def calculate_distance(x, y, z, x_origin=-5, y_origin=24, z_origin=8):
    return math.sqrt((x_origin - x)**2 + (z_origin - z)**2)

def find_random_path(matrix, start):
    rows, cols = len(matrix), len(matrix[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Movements: right, down, left, up

    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)
    previous = {start: None}

    while queue:
        (x, y), dist = queue.popleft()

        if matrix[x][y] == 17:  # Target found
            path = []
            current = (x, y)
            while current is not None:
                path.append(current)
                current = previous[current]
            return path[::-1]  # Reverse path from start to target

        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if (nx, ny) not in visited and matrix[nx][ny] != 2:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
                    previous[(nx, ny)] = (x, y)

    return []  # Return empty list if no path is found

def get_map(level=0):
    x, y, z = get_position()
    sublist_length = 10
    blocks = list(mc.getBlocks(6, y + level, -9, -4, y + level, 0))
    return [blocks[i:i + sublist_length] for i in range(0, len(blocks), sublist_length)]

def find_single_value(matrix, value=1):
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            if element == value:
                return (i, j)
    return None

def step_action(action):
    reward = 0
    done = False
    x0, y0, z0 = get_position()

    if action == 0:
        new_pos = (x0 + 1, y0, z0)
    elif action == 1:
        new_pos = (x0, y0, z0 - 1)
    elif action == 2:
        new_pos = (x0 - 1, y0, z0)
    elif action == 3:
        new_pos = (x0, y0, z0 + 1)

    x, y, z = new_pos
    blocks = list(mc.getBlocks(x - 1, y - 1, z + 1, x + 1, y - 1, z - 1))

    if blocks[4] == 0:
        done = True
        return blocks, reward, done
    elif blocks[4] == 17:
        reward += 10
        done = True
    elif blocks[4] != 0:
        reward += 1

    mc.player.setPos(x, y, z)
    mc.setBlock(x, y - 1, z, 0)

    return blocks, reward, done

def reset():
    print("reset")
    mc.setBlocks(-4, 23, -9, 6, 23, 0, 0)
    x = -3
    y = 24
    z = random.randint(-9, 0)
    mc.setBlock(x, y - 1, z, 2)
    mc.player.setPos(x, y, z)
    z2 = random.randint(-9, 0)
    mc.setBlock(6, y - 1, z2, 17)

    matrix = get_map(level=-1)
    start = find_single_value(matrix, value=2)
    random_path = find_random_path(matrix, start)

    for step in random_path[1:-1]:
        matrix[step[0]][step[1]] = 1

    for i, row in enumerate(matrix[1:], start=1):
        for j, value in enumerate(row):
            if value == 1:
                mc.setBlock(-4 + i, 23, -9 + j, 2)

    blocks = list(mc.getBlocks(x - 1, y - 1, z + 1, x + 1, y - 1, z - 1))
    return blocks

def create_model(input_size, action_size):
    model = tf.keras.Sequential([
        layers.InputLayer(shape=(input_size,)),
        layers.Dense(158),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(0.2),
        layers.Dense(158),
        layers.ELU(),
        layers.Dense(action_size)
    ])
    return model

model = create_model(9, 4)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
losses = []

for episode in range(500):
    state = reset()
    state = np.array([state])
    total_reward = 0

    for t in range(120):
        q_values = model(state)
        action = np.argmax(q_values.numpy())
        next_state, reward, done = step_action(action)
        total_reward += reward

        target = reward + 0.33 * np.max(model(np.expand_dims(next_state, axis=0)).numpy())
        target_q_values = q_values.numpy()
        target_q_values[0, action] = target

        with tf.GradientTape() as tape:
            q_values_pred = model(state)
            loss = loss_fn(target_q_values, q_values_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(loss.numpy())
        state = np.array([next_state])

        if done:
            break

    print(f"Episode: {episode}, Total Reward: {total_reward}, Loss: {loss.numpy()}")

plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Evolution During Training')
plt.show()
model.save('minecraft_agent_6.h5')
