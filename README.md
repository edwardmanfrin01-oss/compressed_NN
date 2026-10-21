GD compression implentated by link/to/github/repo 
The method has been adapted to the specific task of the project. 

_______________________________________________________________**PURPOSE**_________________________________________________________________
Starting from an Image dataset (RAW_IMG), the idea is to obtain a representation (INPUT_IMG) of the compressed file (COMPR_IMG).
The INPUT_IMG is therefore used as input for a neural network model, trying to determine the right class to which the image belongs. 
This process could result as a good starting point for obtaining a faster training and evaluation, reducing the overall parameters of the 
network and also lighter overload of the machine. 

______________________________________________________________**HOW TO USE**_______________________________________________________________

1. DATASET
First of all, select a dataset, and save it in the data folder, for the trials I used CIFAR-10, available from Python libraries for the
download.
If the images are not in the format RGB 224x224, they must be converted, setting the right folders in the script **resize-cifar10.py**.
Then execute it, both for the train and test splits:

    python src/compression/resize-cifar10.py


2. CONVERSION RGB TO YCbCr (OPTIONAL)
For work using the images converted in the YCbCr channels, execute the script for the conversion first. Also here is important to set the 
right paths and directories in the code, before running it:

    python src/dataset_RGB_to_YCbCr.py


3. COMPRESSION 
The compression is obtained through the General Deduplication (GD) approach. The process is iterated over the whole dataset in the script
**batch_compression.py**. It access the Rust source code that implement the compression. It must be executed both for train and test splits
See **.\src\compression\batch_compression.py --help** to see all the commands. To execute (example):

    python .\src\compression\batch_compression.py 
    --input-dir .\data\cifar-10_resized_YCbCr\train 
    --output-dir data/cifar-10_compressed_1x1_YCbCr/train 
    --pixel-grouping 1x1
    --resume

4. FEATURES EXTRACTION
//DESCRIPTION TO WRITE

python src\batch_network_features.py 
--version v3 
--input-dir data\cifar-10_compressed_1x1_YCbCr/train 
--output-dir output\v3_cumulative_rank_adaptive_YCbCr\train 
--resume

