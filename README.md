# camera_to_gridworld

> **_NOTE:_**  Copied From [Tran-Research-Group](https://github.com/Tran-Research-Group)

Pipeline to map a FPV to a top-down gridworld. The model outputs a prediction of whether or not an object is present in a grid-square. Here is a good visualization of the result!

![Model_Output](https://user-images.githubusercontent.com/60635839/167494500-fca2734f-2055-44e1-aea1-8c04dc44f6cf.png)

## Code Structure
```camera_to_gridworld/``` contains:

```jetbot/```: Contains code jetbot uses to collect images.

- ```data/```: Contains images taken by the JetBot that will be used to train the mapper model.

```pipeline/```:

- ```evaluate_model.py```: Loads pretrained pytorch model trained from mapper.py and creates a plot containing (Original Image, True Grid, Predicted Grid). To run, type: python pipeline/evaluate_model.py)

- ```mapper.py```: CNN model for camera to top-down gridworld implementation. To run, type: ```python pipeline/mapper.py```. Can also add optional arguments of batch_size and epochs.

- ```results/```: Contains results from model training and evaluation.

- ```utils.py```: Helper classes and functions for the mapper model.

```resources/```:

- ```papers/```: Notable papers related to the gridworld research project.

- ```issues.md```: Goes over issues encountered during the research and how to fix them as well as help for initial setup for the JetBot.

## Setting up the Environment

Create a new environment using Conda.

```conda create --name gridworld```

Activate the environment you just created.

```conda activate gridworld```

Install the required dependencies.

```
conda install pytorch torchvision -c pytorch
pip install tqdm
conda install scikit-learn
conda install scikit-image
conda install matplotlib

```

## Additional Environment Notes

See resources for more setup, issues, and related papers.
