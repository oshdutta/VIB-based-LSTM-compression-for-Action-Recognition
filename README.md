LSTM Compression with Variational Information Bottleneck
========================================================

#####  This code is for the paper- [A variational information bottleneck based method to compress sequential networks for human action recognition](https://openaccess.thecvf.com/content/WACV2021/html/Srivastava_A_Variational_Information_Bottleneck_Based_Method_to_Compress_Sequential_Networks_WACV_2021_paper.html) published in WACV '21

Required libraries:
-------------------
- pytorch==1.2.0
- torchvision==0.4.0
- numpy=1.19.1
- os
- sys
- argparse
- time
- datetime
- IntEnum

 Reporsitories and previous work:
 -----------------------------------------
- If efficientnet is to be used as a feature extractor, then clone this repository- [Efficient Net](https://github.com/lukemelas/EfficientNet-PyTorch)


Note:
-  ```datasetucf11.py``` may need changes as per your dataset and folder path. 
-  ```dataset_path``` takes the path to the dataset folder; as an example, here ```UCF11_updated_mg-frames``` is the path to the UCF11 dataset
- ```--kml``` takes compression factor multiplier as input. It is kept flexible for each layer which can be modified in ```model_VIBLSTM.py``` 
- To replicate results: VIB pruning can be done in multiple steps- to reduce only LSTM input or feature dimension, set 	```masking=True``` in the definition of ```ib0,ib1,ib2 and ib3``` in ```model_VIBLSTM.py```. To only reduce hidden state dimension, set ```masking=True``` in the definiton of ```ib4``` in ```model_VIBLSTM.py```
		

To compress a CNN-LSTM model on a particular dataset:
-----------------------------------------------------
• To train and prune from scratch:

    python train_VIB.py --dataset_path UCF11_updated_mpg-frames --num_epochs 100 --sequence_length 32 --batch_size 32 --img_dim1 299 --img_dim2 299 --featureExtractor Inc --hidden_dim 2048 --lr 1e-5 --ib_lr 5e-2 --kml 4 --save_dir <directory_name>

• To train and prune a pre-trained CNN-LSTM model:

    python train_VIB.py --dataset_path UCF11_updated_mpg-frames --num_epochs 100 --sequence_length 32 --batch_size 32 --img_dim1 299 --img_dim2 299 --featureExtractor Inc --hidden_dim 2048 --checkpoint_model <optional-path_to_model-to-be-pruned> --lr 1e-5 --ib_lr 5e-2 --kml 4 --save_dir <directory_name>

• To finetune a pruned pre-trained CNN-VIB-LSTM model:

    python train_VIB.py --dataset_path UCF11_updated_mpg-frames --num_epochs 100 --sequence_length 32 --batch_size 32 --img_dim1 299 --img_dim2 299 --featureExtractor Inc --hidden_dim 2048 --finetune_model <optional-path_to_model-to-be-finetune> --lr 1e-5 --ib_lr 5e-2 --kml 4 --save_dir <directory_name>

• To resume pruning a CNN-VIB-LSTM model from a particular checkpoint:

    python train_VIB.py --dataset_path UCF11_updated_mpg-frames --num_epochs 100 --sequence_length 32 --batch_size 32 --img_dim1 299 --img_dim2 299 --featureExtractor Inc --hidden_dim 2048 --resume <optional-path_to_last_model_checkpoint> --lr 1e-5 --ib_lr 5e-2 --kml 4 --save_dir <directory_name>

To compress end-to-end-LSTM model on a particular dataset:
----------------------------------------------------------


• To train and prune end-to-end LSTM model from scratch:

    python train_VIB.py --dataset_path UCF11_updated_mpg-frames --num_epochs 100 --sequence_length 8 --batch_size 128 --img_dim1 160 --img_dim2 120 --e2e True --hidden_dim 256  --lr 1e-5 --ib_lr 5e-2 --kml 2 --save_dir <directory_name>

• To train and prune a pretrained end-to-end LSTM model:

    python train_VIB.py --dataset_path UCF11_updated_mpg-frames --num_epochs 100 --sequence_length 8 --batch_size 128 --img_dim1 160 --img_dim2 120 --e2e True --hidden_dim 256  --checkpoint_model <optional-path_to_model-to-be-pruned> --lr 1e-5 --ib_lr 5e-2 --kml 2 --save_dir <directory_name>

• To finetune a pruned pre-trained end-to-end VIB-LSTM model:

    python train_VIB.py --dataset_path UCF11_updated_mpg-frames --num_epochs 100 --sequence_length 8 --batch_size 128 --img_dim1 160 --img_dim2 120 --e2e True --hidden_dim 256  --finetune_model <optional-path_to_model-to-be-finetuned> --lr 1e-5 --ib_lr 5e-2 --kml 2 --save_dir <directory_name>

• To resume pruning an end-to-end VIB-LSTM model from a particular checkpoint:

    python train_VIB.py --dataset_path UCF11_updated_mpg-frames --num_epochs 100 --sequence_length 8 --batch_size 128 --img_dim1 160 --img_dim2 120 --e2e True --hidden_dim 256 --resume <optional-path_to_last_model_checkpoint> --lr 1e-5 --ib_lr 5e-2 --kml 2 --save_dir <directory_name>


To only evaluate a trained sparse VIB-LSTM(CNN or end-to-end) model:
--------------------------------------------------------------------
Note: Change the image dimensions as per the model

    python train_VIB.py --dataset_path UCF11_updated_mpg-frames --img_dim1 299 --img_dim2 299 --featureExtractor Inc --resume <optional-path_to_model_checkpoint> --save_dir <directory_name> --eval True

To convert the obtained VIB sparse model to a dense-compressed model:
---------------------------------------------------------------------

    python modelport.py --VIBmodel <path_to_VIB_model> --dataset_path UCF11_updated_mpg-frames --latent_dim 2048 --hidden_dim 2048 --featureExtractor Inc
    
### If using this code, kindly cite: 
```
@inproceedings{srivastava2021variational,
  title={A variational information bottleneck based method to compress sequential networks for human action recognition},
  author={Srivastava, Ayush and Dutta, Oshin and Gupta, Jigyasa and Agarwal, Sumeet and AP, Prathosh},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2745--2754},
  year={2021}
}
```
