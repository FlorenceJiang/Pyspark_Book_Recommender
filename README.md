# Overview

The goal of this project is to build and evaluate a realistic and large scale recommendation system using the Goodreads dataset in Spark and Hadoop Distributed File System (HDFS). We used Alternating Least Square algorithm in a model-based collaborative filtering model how hyperparameters "rank" and "regParam" affect the quality of a recommendation system measured by RMSE, precision@k, and mean average precision. 

# Data

We used the [Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) collected by 
> Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018.

# Results

Please refer to our [summary report]() for evaluation results.

# Extensions

We also developed the following extensions:

   - *Comparison to single-machine implementations*: compare Spark's parallel ALS model to a single-machine implementation [lightfm](https://github.com/lyst/lightfm).
  - *Fast search*: use a spatial data structure (LSH) to implement accelerated search at query time. We used an existing library [annoy](https://github.com/spotify/annoy).
  - *Exploration*: use the learned representation to develop a visualization of the items and users. We used [UMAP](https://arxiv.org/abs/1802.03426).
