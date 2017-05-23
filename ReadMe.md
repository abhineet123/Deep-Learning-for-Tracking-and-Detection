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
  * transitions functions are deterministic
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



## Other potentially useful papers
- **Deep Feature Flow for Video Recognition** ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Object%20Detection/Kaiming%20He%2C%20Mask%20R-CNN%2C%202017.pdf), [arxiv](https://arxiv.org/abs/1611.07715), [github](https://github.com/msracver/Deep-Feature-Flow)) by Microsoft Research
  * Summary goes here...

## Datasets
- [traffic-surveillance-dataset collection](https://github.com/gustavovelascoh/traffic-surveillance-dataset) 
- [GRAM Road-Traffic Monitoring](http://agamenon.tsc.uah.es/Personales/rlopez/data/rtm/)
- [Stanford Drone Dataset](http://cvgl.stanford.edu/projects/uav_data/)

## Resources
- [List of deep learning based tracking papers](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-tracking.md)

## Need to decide:
- Data sets
- Papers
- Framework (Fixed to Tensorflow)
