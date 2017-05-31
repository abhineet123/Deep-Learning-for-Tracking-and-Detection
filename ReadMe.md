Collection of papers and other resources for object detection and tracking using deep learning

## Object Detection
- **Mask R-CNN** ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Object%20Detection/Xizhou%20Zhu%2C%20Deep%20Feature%20Flow%20For%20Video%20Recognition%2C%202016.pdf), [arxiv](https://arxiv.org/abs/1703.06870), [github](https://github.com/CharlesShang/FastMaskRCNN)) by Facebook AI Research!
  * Summary goes here...
## Object Tracking
- **Learning to Track: Online Multi-object Tracking by Decision Making** (ICCV 2015) (Stanford) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Learning%20to%20Track%20Online%20Multi-object%20Tracking%20by%20Decision%20Making%20%20iccv15.pdf), [github (Matlab)](https://github.com/yuxng/MDP_Tracking), [project page](https://yuxng.github.io/))
  * RL is used to learn to perform data association between detections and tracked objects;
  * general framework capable of using any existing single object trackers and detectors;
  * the problem of data association between the detector and tracker is cast as an MDP with 4 states (or subspaces of the state space) - active, inactive, tracked, lost
    * active: a new object detected for the first time; 
    * tracked: as long as an active object is still tracked in the current frame
    * lost: active object that is not tracked successfully due to occlusion or temporarily going out of view; can go back to tracked or become inactive;
    * inactive: when an object has been lost for a while; false positive detections also become inactive; permanent state so no transition going out of it
  * transition functions are deterministic
  * reward function is learned using RL to obtain the data association policy for lost state;
  * policy in active sate is learned using SVM - equivalent to learning reward function for that state;
  * policy in lost state is learned using soft margin SVM which gives a similarity measure between the targets and detections which in turn is used with Hungarian algorithm to perform associations between them;
  * learning the similarity function for data association in the lost state - for each target in each training sequence:
    * use the existing policy to take actions as described above;
    * if a target is lost and action does not match the ground truth, add the feature to the training set as negative example if it is a false positive and negative example if it is a false negative; 
      * false positive: target is matched to a detection but is actually not present in the frame
      * false negative: target is not matched to any detection but is actually present in the frame and does match to one of the detections
    * re train the soft margin SVM classifier each time the training set is updated
  * for each new frame:
    * process tracked objects and use non maximum suppression to remove detections covered by tracked objects
    * remove false detections - active to inactive state
    * obtain similarity between all lost targets and detections and use Hungarian algorithm to perform assignment
    * reset template for all assigned targets and move them to tracked state
  * all unassigned detections become new objects and start getting tracked

- **Tracking The Untrackable: Learning To Track Multiple Cues with Long-Term Dependencies** (arxiv April 2017) (Stanford) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Deep%20Learning/Tracking%20The%20Untrackable%20Learning%20To%20Track%20Multiple%20Cues%20with%20Long-Term%20Dependencies%20ax17_4.pdf), [arxiv](https://arxiv.org/abs/1701.01909), [project page](http://web.stanford.edu/~alahi/))
	- learns a representation that encodes long term temporal dependencies between appearance, motion and interaction cues
	- the network eventually provides similarity scores between all 
	targets and detections that are used to arranged these into a bipartite 
	graph that can be subjected to assignment by the Hungarian algorithm;
	- one RNN is used for each cue and outputs from the 3 RNNs are 
	combined by another RNN which outputs a feature vector that in turn is 
	converted into a similarity score;
	- using LSTM to encode long term dependencies means that the 
	similarity score can take into accout a sequence of target patches 
	instead of just the previous one;
	- Appearance cue: 
		- each of the previous t target patches are passed through a CNN which produces a 500-D feature vectors for each;
			- pre trained 16 layer VGGNet with its last FC layer replaced by a 500 sized one;
			- trained using softmax binary classifier

		- all of these (?) are passed through the LSTM that produces an H-D feature vector
		- detection patch is also 
	passed through the CNN and its H-D output feature vector (?) is 
	concatenated with the LSTM output of the last step;
		- the 2H-D concatenated feature
	 vector is passed through an FC layer that finally produces the 500-D 
	feature vector used as the appearance input to the O-RNN

	- Motion Cue:
		- the target velocity in  t frames passed through LSTM to get H-D feature vector
			- target velocity defined as 
	the difference in the x, y coordinates of the bounding box center 
	between the current and previous frames

		- detection velocity passed though FC layer to get another H-D vector that is concatenated with the LSTM one
		- 2H-D vector passed through another FC to get 500-D vector that becomes motion input to the O-RNN;
		-  soft margin classifier used here too;

	- Interaction Cue:
		- a binary occupancy grid is created for each target and represented as a vector
		- each cell of the grid is 1 if at least one of the target’s neighbors is present in it
		- the grids of targets in last t frames are passed through LSTM to get H-D vector
		- grid of detection is passed through FC to get another H-D vector that is concatenated with the last one to get a 2H-D vector
		- this is passed through an FC layer that outputs a 500-D vector that becomes the interaction input to the O-RNN
		- softmax classifier is used here too;

	- Two stage training:
		- the three RNNs are trained separately to output probability of the detection belonging to the trajectory
			- softmax classifier and cross entropy loss used for training

		- joint end to end training of 
	the three RNNs with O-RNN is performed by concatenating their feature 
	vectors (?) and using it as input to the O-RNN
			- the H-D state vector of the 
	last hidden layer of O-RNN  is passed through an FC layer to produce 
	another feature vector (?) that encodes the long term dependencies 
	between the cues
			- this O-RNN is also trained (?) to output a similarity score between the target and detection based on their feature vector

		- both training stages use MOT15 and MOT16 datasets


## Other potentially useful papers
- **Deep Feature Flow for Video Recognition** ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Object%20Detection/Kaiming%20He%2C%20Mask%20R-CNN%2C%202017.pdf), [arxiv](https://arxiv.org/abs/1611.07715), [github](https://github.com/msracver/Deep-Feature-Flow)) by Microsoft Research
  * Summary goes here...

## Datasets
- [traffic-surveillance-dataset collection](https://github.com/gustavovelascoh/traffic-surveillance-dataset) 
- [GRAM Road-Traffic Monitoring](http://agamenon.tsc.uah.es/Personales/rlopez/data/rtm/)
- [Stanford Drone Dataset](http://cvgl.stanford.edu/projects/uav_data/)
- [CBCL StreetScenes Challenge Framework](http://cbcl.mit.edu/software-datasets/streetscenes/)(No top down viewpoint)

## Resources
- [List of deep learning based tracking papers](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-tracking.md)

## Need to decide:
- Data sets
- Papers
- Framework (Fixed to Tensorflow)
