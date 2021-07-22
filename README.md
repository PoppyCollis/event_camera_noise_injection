# event_camera_noise_injection
Augmentation of event-based data using noise injection

The aim of this project is to provide a method by which noise can be injected into a DV event camera dataset for image segmentation with the purpose of improving the robustness of the network's ability to generalize through domain randomization.

The process involves:
1. Converting noise texture videos to events. 
This is done using a docker image (https://hub.docker.com/r/pcoll98/video_to_events_1) of a video-to-event conversion software based on ROS kinetic (see Rebecq, H., Gehrig, D., & Scaramuzza, D. (2018). ESIM: an Open Event Camera Simulator. CoRL: Zurich. http://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf). See video_to_event_instructions.txt.
2. Combining noise events with recorded DV events.
See instructions_for_noise_injection.txt.
 
