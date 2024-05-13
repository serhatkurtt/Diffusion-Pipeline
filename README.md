# Latent Diffusion-Pipeline

In this project, a stable diffusion pipeline is implemented using HuggingFace's Diffusers pipeline framework. Pre-trained variational autoencoders are integrated into the diffusion pipeline to convert the data from image space to latent space. This enables the creation of more realistic images and reduces training time. Then U-net is trained for denoising. Whole pipeline is visualized in the figure below. DDIM (Denoising Diffusion Implicit Models) sampler is used to speed up the sampling process.




  ![](images/Stable_Diffusion_architecture.png?raw=true "Stable diffusion Architecture")

# Dreambooth fine-tuning
Dreambooth is a fine-tuning technique to create personalized images while preserving prior informations learned in training phase to avoid overfitting. Prior-preserving loss function is used in order to prevent overfitting. In this project, Variational autoencoders are also integrated into HuggingFace's Dreambooth-training pipeline. Besides, the goal of the project is unconditional image generation. Hence, text embeddings are removed from the Dreambooth pipeline, and the entire pipeline synthesizes personalized unconditional images.

#Inference 
In order to infer images fro Dreambooth pipeline, data with random distribution is given as input. Then, u-net is used for predicting the noise at each time-step. Then noise eliminated data is converted into image space using variational autoencoder. Finally unconditional personalized images are produced.


