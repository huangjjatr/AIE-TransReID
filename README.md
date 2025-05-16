# Transformer based person re-identification with attribute information embedding

This repository is the official implementation of [Transformer based person re-identification with
attribute information embedding](Submitted to NeurIPS2025). 

## Requirements

To install requirements:

```First setup
pip install -r requirements.txt
(we use /torch 1.6.0 /torchvision 0.7.0 /timm 0.3.2 /cuda 10.1 / 16G or 32G V100 for training and evaluation.
Note that we use torch.cuda.amp to accelerate speed of training which requires pytorch >=1.6)
```

and then install openAI CLIP model:
```install CLIP
pip install git+https://github.com/openai/CLIP.git
```

### Prepare Datasets

Download [Market1501](https://github.com/pseudonymous-aie/AIE-TransReID/blob/main/market1501.zip) and [DukeMTMC](https://github.com/pseudonymous-aie/AIE-TransReID/blob/main/dukemtmc.zip) and then extract file to directory: *AIE-TransReID/datasets*:
```extract Market1501
unzip market1501.zip -d AIE-TransReID/datasets/market1501
```

```extract DukeMTMC
unzip dukemtmc.zip -d AIE-TransReID/datasets/dukemtmcreid
```

### Prepare CLIP and ViT Pre-trained Models
You need to download the pretrained [CLIP ViT-B-16.pt](https://github.com/pseudonymous-aie/AIE-TransReID/blob/main/ViT-B-16.zip)  and then unzip it to directory : *AIE-TransReID/text*.

```CLIP
unzip ViT-B-16.zip -d AIE-TransReID/text
```

You need also to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/pseudonymous-aie/AIE-TransReID/blob/main/jx_vit_base_p16_224-80ecf9dd.zip)  and then unzip it to directory : *AIE-TransReID/model*.
```jx_vit_base
unzip jx_vit_base_p16_224-80ecf9dd.zip -d AIE-TransReID/model
```

## Training
We utilize 1 GPU with 32G GPU memory for training.

To train the model(s) in the paper, in the AIE-TransReID directory run these commands:

```train
./train-market.sh
```
and
```train
./train-duke.sh
```

## Evaluation

After training with attribute information embedding strength varying from 0.1 to 3.0, the models are evaluated automatically and the results are logged into file: *AIE-TransReID/logs/market_attribute/train_bg_384_0.log*. You can change to directory *AIE-TransReID/logs/market_attribute/Market_AIE_SIE.ipynb* and run all the cells to get the curve shown in Fig.1(a) and (c).

```Market1501
cd AIE-TransReID/logs/market_attribute/
jupyter notebook Market_AIE_SIE.ipynb
(after open the notebook in a browser, run all the cells)
```

The training on DukeMTMC dataset using an attribute information embedding strength changing from 0.2 to 3.0. The results shown in Fig.1(b) and (d) can be reproduced by the following code:

```DukeMTMC
cd AIE-TransReID/logs/duke_attribute/
jupyter notebook Duke_AIE_SIE.ipynb
(after open the notebook in a browser, run all the cells)
```
You may also test each trained model with different attribute information embedding strength by using test.py with the settings the same in training.

To reproduce the results shown in Table 1, run test.sh with a command line parameter 'm' first and then 'd' to generated evaluating performance with attribute information dropouts.
```dropout
./test.sh m
./test.sh d
```
Then change to subdirectory : logs, open AIE_test_dropout.ipynb and run all the cells to generate all the data shown in Table 1.
```logs
cd logs
jupyter notebook AIE_test_dropout.ipynb
(after open the notebook in a browser, run all the cells)
```

## Pre-trained Models

The TransReId model uses the pretrained  Google vit_base_p16_224 model. The AIE module uses the OpenAI pretrained CLIP model. Both can be downloaded using the following links:

- [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) trained on ImageNet by Google.
- [ViT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) by OpenAI.

## Results
1. The results of the performance vs. AIE strength &gamma; can be reproduced using our training logs. Goto subdirectory *logs/market_attribute/* and *openMarket_AIE_SIE.ipynb* and run all the cells, you can get the following figures:
![Fig.1(a)](Fig.1(a).png)
![Fig.1(c)](Fig.1(c).png)
Then goto path *logs/duke_attribute/* and open *Duke_AIE_SIE.ipynb*, you can get Fig.1(b) and (d) by running all the cells:
![Fig.1(b)](Fig.1(b).png)
![Fig.1(d)](Fig.1(d).png)
2. The testing results on attribute information dropout with different dropout rate can also be reproduced by our testing results. 
Downloaded the following trained 
* Trained [Market 1501](https://github.com/pseudonymous-aie/AIE-TransReID/blob/main/market_model.zip) with AIE strength &gamma; = 2.6.
```
(after download, under command line run)
unzip market_model.zip -d AIE_TransReID/logs/market_attribute/transformer_0_26.pth
```
* For [DukeMTMC](https://github.com/pseudonymous-aie/AIE-TransReID/blob/main/duke_model.zip) with AIE strength &gamma; = 1.6.
```
(after download, under command line run)
unzip duke_model.zip -d AIE_TransReID/logs/duke_attribute/transformer_0_16.pth
```
Run test.sh will reproduce log files as stated in Evaluation section. Our testing logs are provided so you skip the time-consuming testing. In the *logs* directory, there is an notebook named *AIE_test_dropout.ipynb*. Go to that directory and open it then run all the cells, you'll get those results in Table.1:
#### Attribute dropout results on Market1501 #####
|    | dropout_rate | mAP_mean(%) | mAP_std(%) |  R1_mean(%) | R1_std(%) |
| --- | --- | --- | --- | --- | --- | 
| 0 | 0.01 | 98.11 | 0.13 | 99.52 | 0.07 |
| 1 | 0.02 | 98.11 | 0.16 | 99.57 | 0.08 |
| 2 | 0.05 | 98.12 | 0.14 | 99.51 | 0.11 |
| 3 | 0.1  | 98.13 | 0.10 | 99.56 | 0.07 |
| 4 | 0.2  | 98.11 | 0.13 | 99.53 | 0.08 |

#### Attribute dropout results on DukeMTMC ####
|    | dropout_rate | mAP_mean(%) | mAP_std(%) | R1_mean(%) | R1_std(%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.01 | 93.55 | 0.15 | 97.60 | 0.28 |
| 1 | 0.02 | 93.57 | 0.19 | 97.66 | 0.28 |
| 2 | 0.05 | 93.53 | 0.21 | 97.61 | 0.17 |
| 3 | 0.1  | 93.54 | 0.16 | 97.54 | 0.19 |
| 4 | 0.2  | 93.61 | 0.11 | 97.64 | 0.25 | 

4. Our model achieves the following performance on Market1501 and DukeMTMC datasets with the optimal AIE strength values:

| Datasets     |     mAP    |     Rank-1     |  AIE strength  |
| ------------ | ---------- | -------------- | -------------- |
| Market1501   |     98.50% |      99.60%    |       2.6      |
| DukeMTMC     |     91.2%  |      96.00%    |       1.6      |

The above results can be found in log files: 
1. AIE-TransReID/logs/market_attribute/train_bg_384_0.log (below the line ending with SIE_Lambda: 2.6)
2. AIE-TransReID/logs/duke_attribute/train_384x128_0.log.log (below the line ending with SIE_Lambda: 1.6)


## Contributing

All contributions welcome! 
