# Time normalization

## Code support to a study investigating the effect of time normalization approaches on machine learning classification performance. details about the study can be found in the related [paper](https://doi.org/10.1016/j.jbiomech.2024.112116)

## Pre-requisites

cd to the root folder and run the following commands:
- `conda create -n env tensorflow`
- `conda activate env`
- `conda install --y --file conda_requirements.txt`

## Run the project for the first time (data download and processing)
The necessary data files are already provided but if for some reason they get lost, you can redownload and preprocess them according to the following steps:
- cd to the root folder
- run the command `python main.py --force_data_reload` to download and process the data

## Running the project given the different flag options

- The `--data_type` flag specified the data to be used, it can be set to either `RAW` (by default) for the raw data or `PRO` for the processed data
- The `--normalization_strategy` flag specifies the normalisation approach to be used, it can be set to either `zeropadding` (by default) or `interpolation`
- The `--model_type` flag specifies the modle type to be used during training, it can be set to either `cnn` (by default), `lstm`, or `xgb`
- The `--scaling_strategy` flag specifies the scaling method (`body_mass` or `zero_scaling`)
- The `--filter` flag specifies whether to apply a low pass filter to the data or not

## Using the colab notebooks (no environment configuration needed)
To avoid the environment installation and configuration step, we provide the colab notebooks for each normalization strategy, put the notebooks and the csv files (found in the `data` folder) on you google drive and make sure to specify the correct path to each of the csv files inside the notebooks. for each notebook, run the cells from top to bottom to obtain the required results