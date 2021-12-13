<div align="center"> 
  
# Lab 3 Report

#### Binfang Ye

</div>

<div align="justify"> 
  
In this lab, we need to implement the prune defence that we learnt from the class. The idea is that we prune the convolution layer based on the last pooling average activation arcoss the whole validaiton dataset. Here, we need to prune "conv_3". According to the instruction, we should save the model when the accuracy drops at least {2%, 4%, 10%}. You can find the saved model in this github repo. The name "model_X=2" means the model accuracy drops at least 2%. I also evaluated the attack success rate that is  6.954187234779596% whe the accuracy drops at least 30%. To desgin the goodNet, we need to combine two models together which are Badnet and repaired model. You can find the combined model in code "[MLSec_Lab3.ipynb](https://github.com/yeb2Binfang/ECE-GY-9163-ML-cyber/blob/main/Lab/Lab3/MLSec_Lab3.ipynb)". You do not need to run it from the begining and just start from "Evaluate the combined model" part after you load the original badNet and the dataset. The dataset can be found [here](https://drive.google.com/drive/folders/1Rs68uH8Xqa4j6UxG53wzD0uyI8347dSq). When we prune the model, we should use the clean validation dataset and finally, we should test on the test dataset. 
</div>

<p align="center">
<img src="https://user-images.githubusercontent.com/68700549/145888192-7459abc4-2fc8-47fc-8248-2678847be151.png" align="center" height="300">
</p>

<div align="justify"> 
From the above grpah, we can see that the prune defense is not too successful here because the attack success rate does not drop too much. When the accuracy drop 30%, then attack success rate will drop to 6.954187234779596%. It is ok but not too good because it compromises the accuracy too much. I think the attack method is prune aware attack that the pruned model is retrained with poisoned data. And the weight is changed again so that the model will have the wrong predition. 
</div>
