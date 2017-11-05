Collection of papers and other resources for object detection and tracking using deep learning

## Object Detection
- **Mask R-CNN** ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Object%20Detection/Mask%20R-CNN%20ax17_4.pdf), [arxiv](https://arxiv.org/abs/1703.06870), [github](https://github.com/CharlesShang/FastMaskRCNN)) by Facebook AI Research!
  * Summary goes here...
- Tensorflow object detection API: https://github.com/tensorflow/models/tree/master/object_detection. Only the two SSD nets can run at 12.5 FPS on one GTX 1080 TI (less accurate than YOLO 604x604). Next two models at 4-5 FPS (4-5% mAP better than YOLO). Best model < 1 FPS. Currently code only allow inference of 1 image at a time. Speed might improve by 2.5 times when they allow multiple image inference.
## Object Tracking
- Multi Object Tracking
	- **Learning to Track: Online Multi-object Tracking by Decision Making** (ICCV 2015) (Stanford) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Learning%20to%20Track%20Online%20Multi-object%20Tracking%20by%20Decision%20Making%20%20iccv15.pdf), [github (Matlab)](https://github.com/yuxng/MDP_Tracking), [project page](https://yuxng.github.io/))
	  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/Learning_to_Track_Online_Multi-object_Tracking_by_Decision_Making__iccv15.pdf)

	- **Tracking The Untrackable: Learning To Track Multiple Cues with Long-Term Dependencies** (arxiv April 2017) (Stanford) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Deep%20Learning/Tracking%20The%20Untrackable%20Learning%20To%20Track%20Multiple%20Cues%20with%20Long-Term%20Dependencies%20ax17_4.pdf), [arxiv](https://arxiv.org/abs/1701.01909), [project page](http://web.stanford.edu/~alahi/))
	  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/Tracking_The_Untrackable_Learning_To_Track_Multiple_Cues_with_Long-Term_Dependencies.pdf)

	- **Near-Online Multi-target Tracking with Aggregated Local Flow Descriptor** (ICCV 2015) (NEC Labs) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Multi%20Target/Near-online%20multi-target%20tracking%20with%20aggregated%20local%20%EF%AC%82ow%20descriptor%20iccv15.pdf), [author page](http://www-personal.umich.edu/~wgchoi/))
	  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/NOMT.pdf)
	  
	- **A Multi-cut Formulation for Joint Segmentation and Tracking of Multiple Objects** (highest MT on MOT2015) (University of Freiburg, Germany) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Batch/A%20Multi-cut%20Formulation%20for%20Joint%20Segmentation%20and%20Tracking%20of%20Multiple%20Objects%20ax16_9%20%5Bbest%20MT%20on%20MOT15%5D.pdf), [arxiv](https://arxiv.org/abs/1607.06317), [author page](https://lmb.informatik.uni-freiburg.de/people/keuper/publications.html))
	  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/A_Multi-cut_Formulation_for_Joint_Segmentation_and_Tracking_of_Multiple_Objects.pdf)
	  
- Single Object Tracking
	- **Deep Reinforcement Learning for Visual Object Tracking in Videos** (arxiv April 2017) (USC-Santa Barbara, Samsung Research) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/RL/Deep%20Reinforcement%20Learning%20for%20Visual%20Object%20Tracking%20in%20Videos%20ax17_4.pdf), [arxiv](https://arxiv.org/abs/1701.08936), [author page](http://www.cs.ucsb.edu/~dazhang/))
	  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/Deep_Reinforcement_Learning_for_Visual_Object_Tracking_in_Videos.pdf)
	  
	- **Visual Tracking by Reinforced Decision Making** (arxiv February 2017) (Seoul National University, Chung-Ang University) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/RL/Visual%20Tracking%20by%20Reinforced%20Decision%20Making%20ax17_2.pdf), [arxiv](https://arxiv.org/abs/1702.06291), [author page](http://cau.ac.kr/~jskwon/))
	  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/Visual_Tracking_by_Reinforced_Decision_Making_ax17.pdf)

	- **Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning** (CVPR 2017) (Seoul National University) ([pdf](https://drive.google.com/open?id=0B34VXh5mZ22cZUs2Umc1cjlBMFU), [project page](https://sites.google.com/view/cvpr2017-adnet)) 
	  * [Summary](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/Summary/Action-Decision_Networks_for_Visual_Tracking_with_Deep_Reinforcement_Learning_cvpr17.pdf)

	- **End-to-end Active Object Tracking via Reinforcement Learning** (arxiv 30 May 2017) (Peking University, Tencent AI Lab) ([pdf](https://github.com/abhineet123/Deep-Learning-for-Tracking-and-Detection/blob/master/Tracking/RL/End-to-end%20Active%20Object%20Tracking%20via%20Reinforcement%20Learning%20ax17_5.pdf), [arxiv](https://arxiv.org/abs/1705.10561)

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
- Multi Object Tracking
	- [Learning to Track: Online Multi-Object Tracking by Decision Making (ICCV 2015)](https://github.com/yuxng/MDP_Tracking)[MATLAB]
	- [Multiple Hypothesis Tracking Revisited (ICCV 2015)](http://rehg.org/mht/) (highest MT on MOT2015 among open source trackers)[MATLAB]
	- [Joint Tracking and Segmentation of Multiple Targets (CVPR 2015)](https://bitbucket.org/amilan/segtracking)[MATLAB]
- Single Object Tracking
	- [DeepTracking: Seeing Beyond Seeing Using Recurrent Neural Networks (AAAI 2016)](https://github.com/pondruska/DeepTracking)[Torch 7]
	- [Hierarchical Convolutional Features for Visual Tracking (ICCV 2015)](https://github.com/jbhuang0604/CF2)[Matlab]
	- [Learning Multi-Domain Convolutional Neural Networks for Visual Tracking (Winner of The VOT2015 Challenge)](https://github.com/HyeonseobNam/MDNet)[Matlab/MatConvNet]
	- [RATM: Recurrent Attentive Tracking Model](https://github.com/saebrahimi/RATM)[Python]
	- [Visual Tracking with Fully Convolutional Networks (ICCV 2015)](https://github.com/scott89/FCNT)[Matlab]
	- [Fully-Convolutional Siamese Networks for Object Tracking](https://github.com/torrvision/siamfc-tf)[Tensor flow]
	- [ROLO : Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking](https://github.com/Guanghan/ROLO)[Tensor flow]
	- [Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking (ECCV 2016)](https://github.com/martin-danelljan/Continuous-ConvOp)[MATLAB]
	- [ECO: Efficient Convolution Operators for Tracking (CVPR 2017)](https://github.com/martin-danelljan/ECO)[MATLAB]
	- [End-to-end representation learning for Correlation Filter based tracking (CVPR 21017)](https://github.com/bertinetto/cfnet)[MATLAB]
	- [High-Speed Tracking-by-Detection Without Using Image Information (AVSS 21017)](https://github.com/bochinski/iou-tracker)[Python]
	- [A collection of common tracking algorithms](https://github.com/zenhacker/TrackingAlgoCollection)
- Object Detection and Matching
	- [Matchnet](https://github.com/hanxf/matchnet)
	- [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://github.com/jzbontar/mc-cnn)
	- [Asynchronous Methods for Deep Reinforcement Learning ](https://github.com/miyosuda/async_deep_reinforce)
	- [YOLO9000: Better, Faster, Stronger - Real-Time Object Detection. 9000 classes!](https://github.com/philipperemy/yolo-9000)
	- [Deformable Convolutional Networks](https://github.com/msracver/Deformable-ConvNets)
	- [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://github.com/daijifeng001/R-FCN)
	- [PVANet: Lightweight Deep Neural Networks for Real-time Object Detection](https://github.com/sanghoon/pva-faster-rcnn)
	- [Mask RCNN in TensorFlow](https://github.com/CharlesShang/FastMaskRCNN)
- Optical Flow
	- [Fast Optical Flow using Dense Inverse Search (DIS)](https://github.com/tikroeger/OF_DIS)
	- [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks ](https://github.com/lmb-freiburg/flownet2)
- Misc
	- [Darknet: Convolutional Neural Networks](https://github.com/pjreddie/darknet)
	- [RNNexp](https://github.com/asheshjain399/RNNexp)

	

