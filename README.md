# DESED Classification
Sound Event Classification for DESED Dataset and Activities of Daily Living Dataset with Deep Learning Models (Conv1D, Conv2D, LSTM)
This code is based on [YOUTUBE](https://www.youtube.com/channel/UCEMuqW8dYhhbs4uXtcAaHvQ) video

## Dataset
There are two datasets used in the project

### DESED Dataset
Domestic Environment Sound Event Detection Dataset is provided by [DCASE](http://dcase.community/) for evaluating systems for the detection of sound events using weakly labeled data.
DESED consists of 10 different classes: alarm_bell_ringing, blender, cat, dishes, dog, electric_shaver_toothbrush, frying, running_water, speech, vacuum_cleaner.
You can download DESED from [this website](https://project.inria.fr/desed/).

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119780731-261b5800-bf05-11eb-8900-3472e724afaa.png" width="70%">
</p>

- Soundbank: Foreground and background soundbanks are synthesized and augmented with Scaper to produce synthesized soundscapes
- Synthesized soundscapes: Mixture and foreground and background soundbanks which are strongly labeled
- Recorded soundscapes: Real recorded dataset from [Audioset](https://research.google.com/audioset/index.html) which are unlabeled/weakly labeled/strongly labeled

This project used only strongly labeled data(foreground soundscape, strongly labeled recorded soundscapes) for training.

### Thingy:52 Recorded Dataset
[Thingy:52](https://www.nordicsemi.com/Software-and-tools/Prototyping-platforms/Nordic-Thingy-52) is a multi-sensor prototyping platform including microphone which supports BLE. 
This project collected sounds generated from activities of daily living in real domestic environment with 3 residents with Thingy:52.  
Recorded sounds are annotated with [Audacity](https://www.audacityteam.org/) into 10 different classes: toilet, shower, wash, brush_teeth, dry_hair, cook, eat, wash_dish, watch_tv, vacuum_cleaner.
Each classes contain many sound events for example, toilet class can contain sounds such as toilet flush, fart, and so on. Length of labeled sound data varies from seconds to minutes. 

### Data Integration
DESED dataset and Thingy:52 dataset are integrated for training to classify four different sound events: dishes, frying, running_water, vacuum_cleaner.
Dataset are divided into three groupset for training and integrated Thingy:52 dataset is used for evaluation.
- Soundbank Trained Model: Foreground soundbank is used for training and Thingy:52 recorded dataset is used for evaluation
- Recorded Trained Model: Recorded sounscapes (validation+test) is used for training and Thingy:52 recorded dataset is used for evaluation
- Thingy:52 Trained Model: 80% of Thingy:52 recorded datset is used for training and 20% is used for evaluation

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119785858-95477b00-bf0a-11eb-944c-23e8181269fb.png" width="95%">
</p>

## Data Process
All audio files are downsampled to 16kHz and enveloped with threshold magnitude of 0.003. 
Files are sliced into 1 second delta time and saved in each class directories into samples. 

## Models
Models include Conv1D, Conv2D, and LSTM. 128 log mel-banks are extracted with 25ms window frame and a stride of 10ms.

## Train
Models are selected and trained with training samples. Trained models are saved in `models` directory. 
Accuracy and loss histories are saved in `logs` directory.

## Predict
Predictions are made with evaluation samples and saved in `logs` directory as numpy array. 
Prediction logs are used for confusion matrix. 

### Confusion Matrix
#### Soundbank Trained Model
- Conv1D

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119799366-f4ab8800-bf16-11eb-95dd-f2c9eb2bf010.png" width="50%">
</p>

- Conv2D

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119799493-14db4700-bf17-11eb-8182-b1bdad8afd40.png" width="50%">
</p>

- LSTM

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119799540-1f95dc00-bf17-11eb-8a4f-06e17a0209ea.png" width="50%">
</p>

#### Recorded Trained Model
- Conv1D

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119799833-6aafef00-bf17-11eb-862c-b5172f681d14.png" width="50%">
</p>

- Conv2D

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119799605-30dee880-bf17-11eb-9b5c-ec80ae1625d8.png" width="50%">
</p>

- LSTM

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119799636-389e8d00-bf17-11eb-8702-1eeec7f178f9.png" width="50%">
</p>

#### Thingy:52 Trained Model
- Conv1D

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119800028-98953380-bf17-11eb-98b6-03985dbdb077.png" width="50%">
</p>

- Conv2D

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119799702-49e79980-bf17-11eb-87b3-67f8f6e13996.png" width="50%">
</p>

- LSTM

<p align="center">
  <img src="https://user-images.githubusercontent.com/46836844/119799726-510ea780-bf17-11eb-8db8-42402f0f89eb.png" width="50%">
</p>
