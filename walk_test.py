import tensorflow as tf
from tensorflow.keras import layers  # type: ignore
from mcpi.minecraft import Minecraft
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from time import sleep

# Load the pre-trained model
model = load_model("minecraft_agent_2.h5")

def get_position():
    """Get the player's position in the Minecraft world."""
    player_pos = mc.player.getTilePos()
    x = player_pos.x
    y = player_pos.y
    z = player_pos.z
    return x, y, z

# Initialize Minecraft
mc = Minecraft.create()
mc.postToChat("Program Started")

# Wait for the player to be ready
sleep(30)

# Loop for executing the model's predictions
for i in range(15):
    x0, y0, z0 = get_position()
    
    # Get the blocks around the player
    blocks = np.array([list(mc.getBlocks(x0 - 1, y0 - 1, z0 + 1, x0 + 1, y0 - 1, z0 - 1))])
    
    # Predict the action using the model
    action = model(blocks)
    action = np.argmax(action.numpy())

    # Define the new positions based on the predicted action
    if action == 0:  # Move forward
        new_pos = (x0 + 1, y0, z0)
        print('w')  # Corresponds to "move forward"
    elif action == 1:  # Move left
        new_pos = (x0, y0, z0 - 1)
        print('a')  # Corresponds to "move left"
    elif action == 2:  # Move backward
        new_pos = (x0 - 1, y0, z0)
        print('s')  # Corresponds to "move backward"
    elif action == 3:  # Move right
        new_pos = (x0, y0, z0 + 1)
        print('d')  # Corresponds to "move right"
    
    # Execute the action in Minecraft
    mc.player.setPos(new_pos[0], new_pos[1], new_pos[2])
