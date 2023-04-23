# dvPFT (DDR-based Virtual Pulmonary Function Test) installation instructions.
### Contact: thomas.pisano@pennmedicine.upenn.edu


## Our software uses:
    [Python 3+](https://www.python.org/)
    [SLEAP](https://sleap.ai/): [Pereira et al., Nature Methods, 2022](https://www.nature.com/articles/s41592-022-01426-1).

## Python installation
We suggest using manager like [anaconda](https://www.anaconda.com/download/).

The file in this github repository called: `dvPFT_environment.yml` contains the configuration of the specific anaconda environment used for dvPFT. Install the anaconda environment by running the command:
```
conda env create -f dvPFT_environment.yml
```
This will create an anaconda environment called `dvPFT` on your machine. Activate the environment by running:
```
conda activate dvPFT
```

## SLEAP installation
For most uptodate installation instructions: see [SLEAP installation page](https://sleap.ai/installation.html)
