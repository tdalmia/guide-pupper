# guide-pupper

Guide Pupper: A seeing-eye quadruped robot for the visually impaired. Our robot, named Pupper, is meant to follow a path, detecting visual obstacles and following voice commands along the way. 

Pupper is programed to go forward in a straight line upon detecting an obstacle with its mounted camera. The bounding box of the obstacle is identified, and Pupper will move to the right and around the obstacle. Pupper will vocalize directions via a speaker for the visually impaired person to follow along as it avoids the obstacle. It will then return to the original path. 

Pupper is also equipped with a microphone to listen to voice commands from its owner. Upon hearing 'stop', Pupper will pause its current movement. Upon hearing 'slow', Pupper will reduce its speed by 25%. This way, Pupper's visually impaired owner can help control how Pupper moves and follow along more easily. 

We presented Guide Pupper as our final project for Stanford's CS 123: A Hands-On Introduction to Building AI-Enabled Robots. We thank the course staff for all of the materials and guidance! 

-- Group 2: Tushar Dalmia, Alex Gu, Neha Vinjapuri, Renee Duarte White, Cecelia Wu, Kevin Zhu
