# Grab anything

This is an initial version of the grab anything code.

It uses https://github.com/IDEA-Research/Grounded-Segment-Anything#label-grounded-sam-with-tag2text-for-automatic-labeling (with a more modular script) to recognize the objects using Tag2Text, obtain the pose using groundingdino and segmentate them using SAM.

This code has been developed during MoveIt Hackathon in an interval of 8 hours so is not ready at all for production.

The main idea is to have a way to schedule grasping -> Ideally whisper to command by voice and chat-gpt to disambiguate the sentences and detect which object do the user want to grab.

Then the object can be stored locally and the perception node would segmentate images by freq. In case the "object type" is detected then the algorithm to segementate it is triggered and the planning scene is filled with the mesh of the object.

Then we start manipulation to grab the object following the flow:
- Go closer to the "grasping point".
- Open the gripper.
- Move to grasping point.
- Close the gripper (object should be grasped at this moment).
- Verify grasping.
- Go to dispose location.

For this we can use MoveIt and the new python-bindings to perform the manipulation task.

---
At this moment all the code is encapsulated on a single node but the idea is to divide it by:
- Perception.
- Decision-making or planning.
- Manipulation.

This way we can decouple the logic and perceive the objects while waiting for new commands or performing manipulation.
