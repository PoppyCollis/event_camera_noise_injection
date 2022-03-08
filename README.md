# event_camera_noise_injection
Augmentation of event-based data using noise injection

The aim of this project is to provide a method by which noise can be injected into a DV event camera dataset for image segmentation with the purpose of performing domain randomization to improve the robustness of a network's ability to generalize from training data.

The process involves:
1. Converting noise texture videos to events. 
This is done using a docker image (https://hub.docker.com/r/pcoll98/video_to_events_1) of a video-to-event conversion software based on ROS kinetic (see Rebecq, H., Gehrig, D., & Scaramuzza, D. (2018). ESIM: an Open Event Camera Simulator. CoRL: Zurich. http://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf). See <b>video_to_event_instructions.txt</b>.
2. Combining noise events with recorded DV events.
This is specific to the data collection process outlined in https://github.com/jamesturner246/vicon-dvs-projection which takes the 3D coordinates of the object of interest (in this case, a screwdriver) collected by the vicon camera system, and projects them onto 2D DV space. This means that a dataset in which events of interest are automatically masked and labelled can be collected. See <b>instructions_for_noise_injection.txt</b>.

Perlin noise videos can be found at: https://1drv.ms/u/s!Aki-WPs3PtCniVIiwrK7tp1Kt0N4?e=UHTeT2

Example converted perlin event videos can be found here: https://1drv.ms/u/s!Aki-WPs3PtCnhwth3q2hRuBQEVqY?e=EIIaDw

Example dataset can be found in the folder /noise here: https://1drv.ms/u/s!Aki-WPs3PtCniUzIovfOcF3cbb1M?e=OGs1LV
 
