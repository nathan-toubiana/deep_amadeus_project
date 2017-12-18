# Biaxial Recurrent Neural Network for Music Composition

Collaborative effort from Nicolas Henry, Clément Natta and Nathan Toubiana to generate music from midi files using Biaxial RNNs. This work is based on Daniel Johnson's biaxial model (http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/)


## Structure 

This repo contains one main.ipynb file and 4 folders:


The main.ipynb file contains the model structure, the training process and the prediction generation.

The music_test folder contains the training data (raw midi files)

The model_save folder contains the final model stored

The music_output folder contains final outputs (midi files) to listen to

The func folder contains pieces of code to automate the midi to input conversions (and vice versa) and the tensorboard plug in.



