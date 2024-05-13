# Stable Diffusion-Pipeline

In this project, a stable diffusion pipeline is implemented using HuggingFace's Diffusers pipeline framework. Pre-trained variational autoencoders are integrated into the diffusion pipeline to convert the data from image space to latent space. This enables the creation of more realistic images and reduces training time. Whole pipeline is visualized in the figure below.




  ![](images/Stable_Diffusion_architecture.png?raw=true "Stable diffusion Architecture")


In order to fine-tune stable diffusion models and prevent them from overfitting, several methods are developed. One of these methods is dreambooth-finetuning. Dreambooth is a fine-tuning technique to create personalized images while preserving prior informations learned in training phase to avoid overfitting. In this project, Variational autoencoders are also integrated into HuggingFace's Dreambooth-training pipeline. 


