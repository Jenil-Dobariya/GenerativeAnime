# About Model
This is an image generation model made by implementation of VAE (Variational AutoEncoder). It is trained on the data of images of faces of anime character and the model is able to generate a image when a Normalized tensor of appropriate dimension is given as input to model.

# Use Model
- Clone the repo and run the command:
```
python run.py
```
- image will be generated in generated_img directory. One can change x_, y_ in run.py file to vary number of images.

# Train model
If you want to train model with your data:
- Clone the repo
- Add images from your dataset to images folder
- Install tensorflow, matplotlib, numpy using pip
- Run the command:
```
python train_model.py
```
