# NMA_Project

This repository contains the code from my group's project created and presented during the 2022 Computational Neuroscience course by Neuromatch Academy. We analyzed the *Kay natural images* dataset, sourced from the following [paper](https://www.nature.com/articles/nature06713): Kay, K. N., Naselaris, T., Prenger, R. J., and Gallant, J. L. (2008). Identifying natural images from human brain activity. Nature, 452(7185): 352-355. doi: 10.1038/nature06713. In our project, we programmed a visual encoding model that would predict the fMRI response in 8428 voxels of visual cortex from 1750 natural images.

## Network Architecture

Our visual encoding model implemented ideas from the following [paper](https://doi.org/10.1016/j.jneumeth.2019.108318): Zhang, Chi, et al. “A Visual Encoding Model Based on Deep Neural Networks and Transfer Learning for Brain Activity Measured by Functional Magnetic Resonance Imaging.” Journal of Neuroscience Methods, vol. 325, 2019, p. 108318., https://doi.org/10.1016/j.jneumeth.2019.108318. Our visual encoding model used the pre-trained AlexNet as a fixed feature detector. The EncodingModel.py file contains the code for the model. Our model used the pre-trained AlexNet as a fixed feature detector with a new classifier specific to each region of interest (ROI) in the visual cortex measured in the Kay dataset.

## Results

Our group was successfully able to feed data into the network and output predictions. However, we did not have enough time to debug the network for prediction accuracy to converge to a level close to the reference paper by Zhang et al. (2019).
