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

## RL with QLearning

### Visual Results

#### After Training 20 Epochs
<img src="assets/ql/20.gif" width=400 height=240/>

#### After Training 100 Epochs
<img src="assets/ql/100.gif" width=400 height=240/>

### Test on 100 Epochs
```bash
mean: 225.14527054974315 std: 62.88393475576646 max: 299.080730952846 min: -91.00848452617487
```

## RL with ActorCritic

### Visual Results

#### After Training 200 Epochs
<img src="assets/ac/200.gif" width=400 height=240/>

#### After Training 600 Epochs
<img src="assets/ac/600.gif" width=400 height=240/>

#### After Training 1000 Epochs
<img src="assets/ac/1000.gif" width=400 height=240/>

### Test on 100 Epochs
```bash
mean: 90.56262718486646 std: 50.90599388292423 max: 196.7647751914238 min: -89.969924180479
```

## Offline learning with MLP

### A successful case:
<img src="assets/offline_learning/offline_leanring.gif" width=400 height=240/>

### Test on 100 Epochs
```bash
mean: 228.52481574436288 std: 70.03703811274288 max: 295.4086650330729 min: -167.62464413445562
```

## Reference

1. [https://aayala4.github.io/Lunar-Lander-Python/](https://aayala4.github.io/Lunar-Lander-Python/)
2. [https://zhuanlan.zhihu.com/p/97856004](https://zhuanlan.zhihu.com/p/97856004)
3. [https://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf](https://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf)
4. [https://blog.csdn.net/qq_33302004/article/details/115530428](https://blog.csdn.net/qq_33302004/article/details/115530428)