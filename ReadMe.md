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
	- the network eventually provides similarity scores between all targets and detections that are used to arrange these into a bipartite graph that can be subjected to assignment by the Hungarian algorithm;
	- one RNN is used for each cue and outputs from the 3 RNNs are combined by another RNN which outputs a feature vector that in turn is converted into a similarity score;
	- using LSTM to encode long term dependencies means that the similarity score can take into accout a sequence of target patches instead of just the previous one;
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

- **Near-Online Multi-target Tracking with Aggregated Local Flow Descriptor** (ICCV 2015) (NEC Labs) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Multi%20Target/Near-online%20multi-target%20tracking%20with%20aggregated%20local%20%EF%AC%82ow%20descriptor%20iccv15.pdf), [author page](http://www-personal.umich.edu/~wgchoi/))
  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/NOMT.pdf)

- **Deep Reinforcement Learning for Visual Object Tracking in Videos** (arxiv April 2017) (USC-Santa Barbara, Samsung Research) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/RL/Deep%20Reinforcement%20Learning%20for%20Visual%20Object%20Tracking%20in%20Videos%20ax17_4.pdf), [arxiv](https://arxiv.org/abs/1701.08936), [author page](http://www.cs.ucsb.edu/~dazhang/))
  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/Deep%20Reinforcement%20Learning%20for%20Visual%20Object%20Tracking%20in%20Videos.pdf)
  
- **Visual Tracking by Reinforced Decision Making** (arxiv February 2017) (Seoul National University, Chung-Ang University) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/RL/Visual%20Tracking%20by%20Reinforced%20Decision%20Making%20ax17_2.pdf), [arxiv](https://arxiv.org/abs/1702.06291), [author page](http://cau.ac.kr/~jskwon/))
  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/Visual%20Tracking%20by%20Reinforced%20Decision%20Making%20ax17.pdf)

- **Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning** (CVPR 2017) (Seoul National University) ([pdf](https://drive.google.com/open?id=0B34VXh5mZ22cZUs2Umc1cjlBMFU), [project page](https://sites.google.com/view/cvpr2017-adnet)) 

## Other potentially useful papers
- **Deep Feature Flow for Video Recognition** ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Object%20Detection/Kaiming%20He%2C%20Mask%20R-CNN%2C%202017.pdf), [arxiv](https://arxiv.org/abs/1611.07715), [github](https://github.com/msracver/Deep-Feature-Flow)) by Microsoft Research
  * Summary goes here...

## Datasets
- [GRAM Road-Traffic Monitoring](http://agamenon.tsc.uah.es/Personales/rlopez/data/rtm/)
- [Stanford Drone Dataset](http://cvgl.stanford.edu/projects/uav_data/)
- [Ko-PER Intersection Dataset](http://www.uni-ulm.de/in/mrm/forschung/datensaetze.html)
- [TRANCOS Dataset](http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/)
- [Urban Tracker Dataset](https://www.jpjodoin.com/urbantracker/dataset.html)
- [DARPA VIVID / PETS 2005 dataset](http://vision.cse.psu.edu/data/vividEval/datasets/datasets.html) (Non stationary camera)
- [KIT-AKS Dataset](http://i21www.ira.uka.de/image_sequences/) (No ground truth)
- [CBCL StreetScenes Challenge Framework](http://cbcl.mit.edu/software-datasets/streetscenes/) (No top down viewpoint)
- [MOT 2015](https://motchallenge.net/data/2D_MOT_2015/) (mostly street level camera viewpoint)
- [MOT 2016](https://motchallenge.net/data/MOT16/) (mostly street level camera viewpoint)
- [MOT 2017](https://motchallenge.net/data/MOT17/) (mostly street level camera viewpoint)
- [PETS 2009](http://www.cvg.reading.ac.uk/PETS2009/a.html) (No vehicles)
- [PETS 2017](https://motchallenge.net/data/PETS2017/) (Low density; mostly pedestrians)
- [KITTI Tracking Dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) (No top down viewpoint; non stationary camera)

## Resources
- [List of traffic surveillance datasets](https://github.com/gustavovelascoh/traffic-surveillance-dataset) 
- [List of deep learning based tracking papers](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-tracking.md)
- [List of multi object tracking papers](http://perception.yale.edu/Brian/refGuides/MOT.html)
- [List of single object trackers with results on OTB](https://github.com/foolwood/benchmark_results)
- [List of Matlab frameworks, libraries and software](https://github.com/uhub/awesome-matlab)

## Tutorials
- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
- [Demystifying Deep Reinforcement Learning](https://www.intelnervana.com/demystifying-deep-reinforcement-learning/)

## Code
- [Learning to Track: Online Multi-Object Tracking by Decision Making (ICCV 2015)](https://github.com/yuxng/MDP_Tracking)
- [Matchnet](https://github.com/hanxf/matchnet)
- [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://github.com/jzbontar/mc-cnn)
- [Asynchronous Methods for Deep Reinforcement Learning ](https://github.com/miyosuda/async_deep_reinforce)
- [YOLO9000: Better, Faster, Stronger - Real-Time Object Detection. 9000 classes! ](https://github.com/philipperemy/yolo-9000)
- [Deformable Convolutional Networks](https://github.com/msracver/Deformable-ConvNets)
- [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://github.com/daijifeng001/R-FCN)
- [Darknet: Convolutional Neural Networks](https://github.com/pjreddie/darknet)
- [PVANet: Lightweight Deep Neural Networks for Real-time Object Detection](https://github.com/sanghoon/pva-faster-rcnn)
- [Fast Optical Flow using Dense Inverse Search (DIS)](https://github.com/tikroeger/OF_DIS)
- [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks ](https://github.com/lmb-freiburg/flownet2)
- [RNNexp](https://github.com/asheshjain399/RNNexp)
- [A collection of common tracking algorithms](https://github.com/zenhacker/TrackingAlgoCollection)
- [Multiple Hypothesis Tracking Revisited (ICCV 2015)](http://rehg.org/mht/) (highest MT on MOT2015 among open source trackers)
- [Joint Tracking and Segmentation of Multiple Targets (CVPR 2015)](https://bitbucket.org/amilan/segtracking)
- [DeepTracking: Seeing Beyond Seeing Using Recurrent Neural Networks (AAAI 2016)](https://github.com/pondruska/DeepTracking)
- [Hierarchical Convolutional Features for Visual Tracking (ICCV 2015)](https://github.com/jbhuang0604/CF2)
- [Learning Multi-Domain Convolutional Neural Networks for Visual Tracking (Winner of The VOT2015 Challenge)](https://github.com/HyeonseobNam/MDNet)
- [RATM: Recurrent Attentive Tracking Model](https://github.com/saebrahimi/RATM)
- [Visual Tracking with Fully Convolutional Networks (ICCV 2015)](https://github.com/scott89/FCNT)
- [Fully-Convolutional Siamese Networks for Object Tracking (Tensor flow)](https://github.com/torrvision/siamfc-tf)
- [ROLO : Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking](https://github.com/Guanghan/ROLO)
- [Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking (ECCV 2016) (MATLAB)](https://github.com/martin-danelljan/Continuous-ConvOp)
- [ECO: Efficient Convolution Operators for Tracking (CVPR 2017)](https://github.com/martin-danelljan/ECO)
- [End-to-end representation learning for Correlation Filter based tracking (CVPR 21017)](https://github.com/bertinetto/cfnet)
