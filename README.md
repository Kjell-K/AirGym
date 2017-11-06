UPDATE: 
- New reset() integrated
- Better Train / Test structure
- Updated binaries for Win 10
- Better reward function

# AirGym
This repository integrates AirSim with openAI gym and keras-rl for autonomous copter through reinforcement learning.

The integration to gym is adapted from [DRL-AutonomousVehicles](https://github.com/kaihuchen/DRL-AutonomousVehicles) and extended by the multirotor integration.

Requirements:

[AirSim](https://github.com/Microsoft/AirSim)

[keras-rl](https://github.com/matthiasplappert/keras-rl)

[openAI gym](https://github.com/openai/gym)


My test environment binaries for Win10 can be downlaoded [here](https://drive.google.com/open?id=0ByG_CWp-MUNNTjg2cllsMGhRbDg).

Click here for a demo video:

[![Youtube Video here](https://img.youtube.com/vi/ZE5hPHqJC64/0.jpg)](https://youtu.be/ZE5hPHqJC64)

#### How to use:
You can either train yourself or load the exciting weights by setting Train to True or False. 
CAREFUL: When you cancel the training with STRG + C, weights are saved and will overright the already trained weights.

#### Status:
Right now the framework has proofed to be able to learn, with 3 score inputs. The score inputs take the average of the left, middle and right section of the depth image respectively. 

#### Next step: 
Be able to learn with full raw depth image. (For right now I am taking a screen shot of the DepthVis window of AirSim, since the API to receive the DepthVis image returns false data.)

#### Issues:
How to pass additional information beside the image. (While using the score values, there is no difficulty to pass more information like orientation or distance from goal in the gym.Box as well)

