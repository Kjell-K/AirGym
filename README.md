Note: This repository is not maintained any longer and is very likely that new versions of AirSim will not be compatible (e.g. methods of AirSim now require different amount of inputs etc.). 

However, from this project you can still retrace how to expose AirSim as an OpenAI Gym environment.

If somebody updates for the newest AirSim, I would be happy for a pull request.

___________________________________________

# AirGym
This repository integrates AirSim with openAI gym and keras-rl for autonomous vehicles through reinforcement learning. AirSim allows you to easly create your own environment in Unreal Editor and keras-rl let gives your the RL tools to solve the task. 

Requirements:

[AirSim](https://github.com/Microsoft/AirSim)

[keras-rl](https://github.com/matthiasplappert/keras-rl)

[openAI gym](https://github.com/openai/gym)


My test environment binaries for Win10 can be downlaoded [here](https://drive.google.com/open?id=1iNeK47r9e54Ba554rHY8o5ADeiDVyPLz).

Click here for a old (the performance is much better now) demo video:

[![Youtube Video here](https://img.youtube.com/vi/ZE5hPHqJC64/0.jpg)](https://youtu.be/ZE5hPHqJC64)

#### How to use:
You can either train yourself or load the exciting weights by setting Train to True or False. 
CAREFUL: When you cancel the training with STRG + C, weights are saved and will overright the already trained weights.

#### State:
We are taking as state input a depth image extended by the encoded information of the relative goal direction. Take a look at it by uncommenting [here](https://github.com/Kjell-K/AirGym/blob/master/gym_airsim/envs/myAirSimClient.py#L155). 

#### Action:
For this environment, we force the quadcopter to move in a fix plane and therefore confront the obstacles. The action space consist of three discrete actions and are available at any state:
- straight: Move in direction of current heading with 4m/s for 1s
- right yaw: Rotate right with 26°/s for 1s
- left yaw: Rotate left with 30°/s for 1s
