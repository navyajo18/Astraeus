# Astraeus
OUR PROJECT 

<p>
  <img src="./app/static/images/logo.png" alt="Astraeus logo" width="64" />
</p>

Problem: The problem that we aim to solve with Astraeus is the ability to identify different celestial beings in the night sky from camera footage to help major space companies with research and identification.  

This technology can be used to help map how much the universe is expanding!

Strategic Aspects: Strategically speaking, this project highlights how machine learning can transform the field of astronomy by automating the classification of vast celestial datasets that are too large for manual analysis. In the long term, it positions the research within a growing global effort to apply AI for large-scale scientific discovery and space data analysis.

Relation to Course Lectures/Papers: We will be using regularization to prevent overfitting in our dataset. Additionally, our multiclass classification will be done by a convolutional neural network (CNN). Our optimizer, Adam, is an adaptive form of gradient descent.

NOVELTY AND IMPORTANCE 

Why our project is important: This project is important because it applies machine learning to automate celestial image classification, a process that traditionally requires extensive human expertise and time. As astronomical datasets continue to evolve due to advances in space exploration missions, manual sorting and identification of celestial bodies have become increasingly impractical. Automating this task with CNNs enables faster and more consistent classification, allowing astronomers to focus on deeper scientific analysis rather than data labeling.

Existing Issues: Existing issues include the sheer volume and imbalance of available data, where some celestial classes (like stars) are overrepresented while others (like comets or black holes) are rare. Additionally, image quality can vary significantly due to differences in exposure, telescope sensors, and atmosphere conditions, which only makes accurate classification more challenging. 

Prior Related Works: Prior works have explored similar approaches, such as NASA’s Galaxy Zoo project, which aimed to use crowdsourcing labeling combined with neural networks to classify galaxies. Recent studies have also tried to apply deep learning to Hubble and Sloan Digital Sky Survey (SDSS) data. This demonstrates the potential of CNN-based methods in astronomy but also showcases ongoing challenges in generalizing models across diverse, heterogeneous, and noisy image datasets (hence we will have to use regularization).

PLAN 

Dataset: Our dataset, titled “SpaceNet: A Comprehensive Astronomical Dataset” encompasses a wide variety of celestial bodies, offering approximately 12,900 samples. The dataset is split into 8 classes  including planets, galaxies, asteroids, nebulae, comets, black holes, stars, and constellations that we can use to train our machine learning model. This split will help ensure that our dataset is not overwhelmingly large, and also ensure proper fitting. Additionally, to supplement the original dataset, a second dataset, titled “Digital Modulation Constellation Images” will be joined.

How to obtain: This dataset is extremely large and cannot be manually uploaded to our IDEs, so we will be storing images by designing SQL databases using BLOB (Binary Large Object) data types that will be split into 8 different tables (representing each class). We will then clean and prepare the data by handling missing or corrupted images. 

Created/simulated: This dataset was simulated using real astronomical images. The directory is hierarchically structured, providing high-resolution (HR) images that have undergone a series of augmentations and synthetic sample generation using advanced techniques using the FLARE project’s novel double stage augmentation strategy.

Models/Implementation: After storing the SpaceNet images in SQL databases as BLOBs separated into eight celestial classes, the next step will involve retrieving and preprocessing the data for model training. Each image will be queried, decoded into a numerical array, resized to a uniform 128x128 dimension, and then normalized so that pixel values range between 0 and 1. The labels representing each of the celestial body types will be numerically encoded, and the dataset will be split into training and testing. 

A Convolutional Neural Network (CNN) will then be designed for multiclass classification. Since the dataset includes eight categories, a softmax activation function will be used in the final layer to produce a probability distribution over all classes, allowing the model to predict which celestial type is most likely. The CNN will include many convolutional and pooling layers to extract spatial features, and will then be followed by fully connected dense layers that learn complex patterns. We will also be using the class concept of regularization to prevent overfitting. 

Techniques
The model will be trained with the Adam optimizer, which is an adaptive form of gradient descent that adjusts learning rates for each parameter to speed up convergence. The loss function (which will be categorical cross-entropy) will be minimized through gradient descent updates. Throughout training, accuracy and loss will be tracked and visualized using Matplotlib, which will then be used to evaluate  convergence and model stability. 
    Gradient Descent Optimization 

Evaluation: Matplotlib will be used extensively to visualize the model’s performance over time, providing clear insight into how the CNN learns during training. By plotting metrics such as training and validation loss, accuracy, and convergence curves, it will truly help identify whether the model is overfitting, underfitting, or steadily getting better.

Additional Features: We may also incorporate computer vision to enhance user experience.

CITATIONS:
In 2024, Mohammed Talha Alam, Raza Imam, Mohsen Guizani, and Fakhri Karray released a paper titled FLARE up your data: Diffusion-based Augmentation Method in Astronomical Imaging. It was published on arXiv in the computer vision category with the identifier 2405.13267.} 
Link to Dataset: https://www.kaggle.com/datasets/razaimam45/spacenet-an-optimally-distributed-astronomy-data 
