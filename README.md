# Latent Diffusion-Pipeline

In this project, a stable diffusion pipeline is implemented using HuggingFace's Diffusers pipeline framework. Pre-trained variational autoencoders are integrated into the diffusion pipeline to convert the data from image space to latent space. This enables the creation of more realistic images and reduces training time. Then U-net is trained for denoising and. Whole pipeline is visualized in the figure below.




  ![](images/Stable_Diffusion_architecture.png?raw=true "Stable diffusion Architecture")

# Dreambooth fine-tuning
Dreambooth is a fine-tuning technique to create personalized images while preserving prior informations learned in training phase to avoid overfitting. Prior-preserving loss function is used in order to prevent overfitting. In this project, Variational autoencoders are also integrated into HuggingFace's Dreambooth-training pipeline. Besides, the goal of the project is unconditional image generation. Hence, text embeddings are removed from the Dreambooth pipeline, and the entire pipeline synthesizes personalized unconditional images.


