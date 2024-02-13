# dPFT (DDR-based Pulmonary Function Test)

![plot](./static/dPFT_pipeline.png)

dPFT ([Santibanez, V., Pisano, T.J., et al. 2024](https:....)) are generated using an automated lung analysis pipeline that takes raw dynamic digital radiography (DDR) videos and outputs DDR-based Pulmonary Function Test (dPFT) data. This is accomplished using convolutional neural networks for serial anatomical detection across frames.

An overview of dPFT is shown here:

![](./static/dPFT_pipeline.gif)

## Installation

Please see [INSTALLATION.md](INSTALLATION.md) for installation instructions.

## Example use cases and tutorials

Please see [dPFT_example.ipynb](dPFT_example.ipynb) for basic dPFT inference and analysis example.

Trained neural networks can be downloaded below:
[All models (recommended)](https://drive.google.com/file/d/1G6EONII9j-104F6WON_uIb8BVnREKFxc/view?usp=sharing).
[Individual models](https://drive.google.com/drive/folders/1I9StYU17ylPY6CnmhpdHt6q0eR8J6LAk?usp=drive_link).
[Annotated SLEAP training sets, to use for transfer learning (Advanced)](https://drive.google.com/drive/folders/14ain4SmGmawfU6okkvwAG_0ZgSLEdWUD?usp=drive_link).
