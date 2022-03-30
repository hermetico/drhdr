# DRHDR

## Setup

Install the dependencies with pip

```
pip install -r requirements.txt

```

Download the candidate [model](https://drive.google.com/file/d/1c3JKQ2sztLWxih-Hh5ec9neM072y3m_f/view?usp=sharing):

```
mkdir -p checkpoints/candidate/models/
cd checkpoints/candidate/models/
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1c3JKQ2sztLWxih-Hh5ec9neM072y3m_f" -O "best_checkpoint.pth"
```



To produce the final results run:

```
sh launch-inference.sh
```

To compute only the metrics run:

```
sh launch-metrics.sh
```