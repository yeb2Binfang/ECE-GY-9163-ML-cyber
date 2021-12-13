<div align="center"> 
  
# Lab 3 Report

#### Binfang Ye

</div>

<div align="justify"> 
  
In this lab, we need to implement the prune defence that we learnt from the class. The idea is that we prune the convolution layer based on the last pooling average activation arcoss the whole validaiton dataset. Here, we need to prune "conv_3". According to the instruction, we should save the model when the accuracy drops at least {2%, 4%, 10%}. You can find the saved model in this github repo. The name "model_X=2" means the model accuracy drops at least 2%. To desgin the goodNet, we need to combine two models together which are Badnet and repaired model. You can find the combined model in code "[MLSec_Lab3.ipynb](https://github.com/yeb2Binfang/ECE-GY-9163-ML-cyber/blob/main/Lab/Lab3/MLSec_Lab3.ipynb)". You do not need to run it from the begining and just start from "Evaluate the combined model" part.  
</div>
