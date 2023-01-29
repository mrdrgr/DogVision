## Introduction
"Dog Vision" is a learning project that uses deep learning models to classify images of dog breeds from 120 dog breed classes. The project is built using tranfser learning techniques and can predict dog breeds from images with high accuracy level. All the trained models are stored in the models and ready to use. The demo folder is created in order to deploy the project to Hugging Face for live demo.  

## Prerequisites
Python  
Pytorch  
Matplotlib  
Tqdm  
Gradio  

## Usage Guide
'main.ipynb': Containing all the code for model creation and training.  
'classes.txt': An structured file containing the list of all different classes.  
'models': All of trained models are stored in this folder.  
'runs': This folder is created in order to store Tensorboard logs.  
'demo': Contains all is needed to make a live demo using Gradio and Hugging Face.  

## Deployment
The porject demo folder can be deployed using Gradio and Hugging Face to create a live demo.  

## Built With
Pytorch - The deep learning framework used  
Tensorboard - Tracking and comparing performances of models  
Gradio - Used for model deployment  
