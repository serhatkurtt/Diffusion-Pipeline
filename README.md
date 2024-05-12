# Diffusion-Pipeline

In this project, a stable diffusion pipeline is implemented using HuggingFace's Diffusers pipeline framework. Pre-trained variational autoencoders are integrated into the diffusion pipeline to convert the data from image space to latent space. This enables the creation of more realistic images and reduces training time. Whole pipeline is visualized in the figure below.




  ![](images/Stable_Diffusion_architecture.png?raw=true "Stable diffusion Architecture")


In order to fine-tune stable diffusion models and prevent them from overfitting, several methods are developed. One of these methods is dreambooth-finetuning. Dreambooth is a fine-tuning technique to create personalized images while preserving prior informations learned in training phase to avoid overfitting. In this project, Variational autoencoders are also integrated into HuggingFace's Dreambooth-training pipeline. 

DEPENDENCIES

pip install diffusers==0.20.2
pip install accelerate
pip install datasets
pip install bitsandbytes

In order to train the stable diffusion pipeline, following command is given as input:

!python diffusion_model.py \
--output_dir  "path/to/output/model" \
--train_data_dir  "path/to/traindata/" \
--mixed_precision "fp16"\
--resolution  256\
--prediction_typ "epsilon"

In order to run Dreambooth finetuning, following command is given as input:

!python Dreambooth_training.py \
--pretrained_model_name_or_path "path/to/trained_model" \
--instance_data_dir="path/to/instancedata_dir" \
--class_data_dir="/path/to/classdatadir" \
--output_dir="/path/to/output/model/"


In order to infer from the fine-tuned model:

!python InferenceScript.py \
--out_name "path/to/outputdir.png" \
--pipeline_dir "path/to/pipeline"
