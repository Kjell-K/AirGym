# AirGym
This repository integrades AirSim with openAI gym and keras-rl for autonomous copter through reinforment learning.

It is based on the work of [DRL-AutonomousVehicles](https://github.com/kaihuchen/DRL-AutonomousVehicles).

Requirements:

[AirSim](https://github.com/Microsoft/AirSim)

[keras-rl](https://github.com/matthiasplappert/keras-rl)

[openAI gym](https://github.com/openai/gym)


My test environment binaries for Win10 can be downlaoded [here](https://drive.google.com/open?id=0ByG_CWp-MUNNNzh0UVowcVk2OVk).

![](https://github.com/Kjell-K/AirGym/blob/master/Results/First_Train.gif)

Status:
Right now the framework has proofed to be able to learn, with 3 score inputs. The score inputs take the average of the left, middle and right section of the depth image respectively. 

Next step: 
Be able to learn with full raw depth image. 

Issues:
How to pass additional information beside the image. (While using the score values, there is no difficulty to pass more information like orientation or distance from goal in the gym.Box as well)

