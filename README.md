# SNGAN
## Simplistic implementation of spectral normalization for GANs.

### Motivation

I had difficulty understanding how to implement SN GAN and was not able resources that does explain the implementation. Hence here I provide a very simplistic implementation on MNIST with only fully connected layers.

### How to run

    python main.py train [options] 
    
### Options

    --img_size=28           :   Size of the images, for MNIST its 28x28
    --channels=1            :   Number of channels in an image, for MNIST its greyscale images
    --data_folder=data      :   Folder to store the dataset
    --samples_folder=data   :   Folder to store the samples generated during training
    --batch_size=128        :   Batch size during training
    --latent_dim=100        :   Size of latent vector that is fed to the generator
    --n_cpu=12              :   Number of cpu threads to allocate for data processing
    --n_critic=5            :   Number of iteration to train the discriminator per every one iteration training for generator
    --lr=0.01               :   Learning-rate of both discriminator and generator, the larger the batch size, the bigger this number can be
    --betas=(0.5, 0.9)      :   Adam optimizers beta hyperparameters
    --n_epochs=200          :   Maximum number of epochs to train
    --sample_interval=500   :   Generate samples from generator every `sample_interval` iteration during training.
    