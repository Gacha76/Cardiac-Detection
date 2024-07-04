## Introduction
In this project we will preprocess the data for the cardiac detection task. Then, we will create a custom DataSet which will load and return an X-Ray image together with the location of the heart.

At first we download the data from kaggle (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)

Dataset:
Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017, http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf

Some example images from dataset :<br/> ![output](https://github.com/Gacha76/Cardiac-Detection/assets/114499152/684f739f-88e7-4798-abec-898c68346bf8)

We provide bounding boxes for around 500 images of the RSNA pneumonia detection challenge dataset.

New Dataset: 
<table width="100%"> 
<tr>
<td width="50%">      
&nbsp; 
<br>
<img src="https://github.com/Gacha76/Cardiac-Detection/assets/114499152/caf1a4ba-ee09-4001-81d4-b2c420fb1fb1">

</td> 
<td width="50%">
<br>
<img src="https://github.com/Gacha76/Cardiac-Detection/assets/114499152/09b979fd-2b4c-4ee1-b921-dccd1beee473">

</td>
</tr>

<tr>
<td width="50%">      
&nbsp; 
<br>
<img src="https://github.com/Gacha76/Cardiac-Detection/assets/114499152/304ad79f-89b4-4e4b-9501-986afc3d137e">


</td> 
<td width="50%">
<br>
<img src="https://github.com/Gacha76/Cardiac-Detection/assets/114499152/014f3b17-9822-4cef-b7b2-63056950c1a0">


</td>
</tr>
</table>

We will again convert the images to npy files for efficient storage and faster data loading.

In order to efficiently handle our data in the Dataloader, we convert the X-Ray images stored in the DICOM format to numpy arrays. Afterwards we compute the overall mean and standard deviation of the pixels of the whole dataset, for the purpose of normalization.
We standardize all images by the maximum pixel value in the provided dataset, 255.
All images are resized to 224x224.
To compute dataset mean and standard deviation, we compute the sum of the pixel values as well as the sum of the squared pixel values for each subject.
This allows to compute the overall mean and standard deviation without keeping the whole dataset in memory.

## Network architecture: Resnet18
As most of the torchvision models, the original ResNet expects a three channel input in **conv1**. <br />
However, our X-Ray image data has only one channel.
Thus we need to change the in_channel parameter from 3 to 1.

4 outputs: We need to estimate the location of the heart (xmin, ymin, xmax, ymax).

Loss function: We are going to use the L2 loss (Mean Squared Error), as we are dealing with continuous values.

After applying data augmentation and normalization: <br/> ![Screenshot 2024-07-04 212516](https://github.com/Gacha76/Cardiac-Detection/assets/114499152/90199ba6-9acc-4879-9a6e-d752ca333ae3)


Final result: <br/> ![Screenshot 2024-07-04 212547](https://github.com/Gacha76/Cardiac-Detection/assets/114499152/0132c3df-be9c-4dbf-9e20-467e6e4a3157)
