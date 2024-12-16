from mcpi.minecraft import Minecraft
import json
import pyautogui
from time import sleep

# Open the JSON file
with open("resultados.json", "r") as f:
    # Load the content of the JSON file as a dictionary
    data = json.load(f)

mc = Minecraft.create()
sleep(30)
mc.postToChat(f"Program Started")
c = 0

while True:
    pos_player = mc.player.getTilePos()  # Get the player's position
    # Make the position relative to absolute
    x = pos_player.x + 8.5
    y = pos_player.y + 64
    z = pos_player.z + 229
    print(x, y, z)
    
    # Save the position relative to the spawn (8.5, 64, 229)
    x_r = pos_player.x
    y_r = pos_player.y
    z_r = pos_player.z
    
    # Get the 9 blocks below the player
    blocks = mc.getBlocks(x_r - 1, y_r - 1, z_r + 1, x_r + 1, y_r - 1, z_r - 1)
    
    # Position 4: block where the player is
    # Position 7: block in front of the player below
    blocks = list(blocks)
    
    # If there's a block in front, move forward
    if c > 20:
        break
    if blocks[7] != 0:
        pyautogui.keyDown("w")
        sleep(0.13)
        pyautogui.keyUp("w")
    elif blocks[3] != 0:
        pyautogui.keyDown("a")
        sleep(0.13)
        pyautogui.keyUp("a")
    elif blocks[5] != 0:
        pyautogui.keyDown("d")
        sleep(0.13)
        pyautogui.keyUp("d")
    else:
        break
    sleep(2)
    c += 1
