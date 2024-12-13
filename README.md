# guide-pupper

## Description 

Guide Pupper: A seeing-eye quadruped robot for the visually impaired. Our robot, named Pupper, is meant to follow a path, detecting visual obstacles and following voice commands along the way. 

Pupper is programed to go forward in a straight line upon detecting an obstacle with its mounted camera. The bounding box of the obstacle is identified, and Pupper will move to the right and around the obstacle. Pupper will vocalize directions via a speaker for the visually impaired person to follow along as it avoids the obstacle. It will then return to the original path. 

Pupper is also equipped with a microphone to listen to voice commands from its owner. Upon hearing 'stop', Pupper will pause its current movement. Upon hearing 'slow', Pupper will reduce its speed by 25%. This way, Pupper's visually impaired owner can help control how Pupper moves and follow along more easily. We use OpenAI's Whisper for automatic speech recognition and then detect whether one of these keywords was present in the recorded and transcribed audio. 

We presented Guide Pupper as our final project for Stanford's CS 123: A Hands-On Introduction to Building AI-Enabled Robots. We thank the course staff for all of the materials and guidance! 

-- Group 2: Tushar Dalmia, Alex Gu, Neha Vinjapuri, Renee Duarte White, Cecelia Wu, Kevin Zhu

## Dependencies & Instructions

Our main code for the project can be found in guide_pupper.py. 

To run our code, you can use the following steps: 
* SSH into Pupper. 
* Install pygame, sounddevice, and openai-whisper via pip.
* Download Foxglove to your desktop and connect to Pupper.
* Pair a microphone and transmitter to Pupper.
* Add an OpenAI Whisper API key into guide_pupper.py line 43. 
* Run the run.sh file.
* Run the guide_pupper.py file. 

Upon executing the run.sh file, you should see an image of what Pupper sees in Foxglove. Pupper's motors should also be in their default position with some resistance, ready to walk. 

Upon running the guide_pupper.py file, Pupper will begin to walk straight until detecting an obstacle, from which it will reroute. You can speak into the microphone when the terminal logs show "Recording audio...". 



