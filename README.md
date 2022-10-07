# POET_Maze

This project was for a class I took on reinforcement learning. I created my own maze reinforcement 
learning environement that randomly generates mazes of a given size using Kruskal's algorithm. What 
I wanted to test was whether I could get multiple reinforcement learning agents stuck in a maze to
communicate to more efficiently solve it. Namely, the maze is only complete once all agents have 
reached the end, and I allowed the agents to display different numbers and change their number 
representation. Communictaion could help them solve it by, for example, every agent dispersing, 
coming back, and communicating to see if anyone found the end. Or, one agent could stay behind and
be an oracle that gives directives to others on where to go and whether the end has been found. To 
discover solutions such as these, I implemented the paired open-ended trailblazer algorithm (POET),
which spawns environment-agent pairs of increasing environment difficulty as the algorith 
progresses. It also allows agents to transfer environments under specific criteria. I used the size
of the maze as a proxy for it's difficulty, and ensured that the start was always at least half the
size of the maze away from the end. For the reinforcement learning agents, I used PyTorch's LSTM 
cell, as the agents need some memory of where they've been. Additionally, they have limited field 
of vision, which is further blocked by the walls, like being in a real maze. The POET algorithm is 
quite computationally expensive, and so I was only able to train the agents to solve mazes up to a
size of 4x5, and was not able to find evidence of communication.

Below is a video of the agents solving a 3x3 maze. The yellow square around the agents is their field
of vision, and their colour is their representation. The red square is the starting location and 
the green square is the end of the maze.

https://user-images.githubusercontent.com/38572823/194663903-850d93bd-a944-42b6-a76c-7b74978c6628.mp4
