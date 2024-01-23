# AI Project: Lunar Lander

## Set up env

```bash
conda create -n lunar python=3.8
conda install swig
pip install gym[box2d]
# install pytorch you need
conda install -c ankurankan pgmpy # may be for Probabilistic Graphical Models
pip install tensorboard tensorboardX
```

## RL with ActorCritic

### Visual Results

#### After Training 200 Epochs
<img src="assets/ac/200.gif" width=400 height=240/>

#### After Training 600 Epochs
<img src="assets/ac/600.gif" width=400 height=240/>

#### After Training 1000 Epochs
<img src="assets/ac/1000.gif" width=400 height=240/>


## Reference

1. [https://aayala4.github.io/Lunar-Lander-Python/](https://aayala4.github.io/Lunar-Lander-Python/)
