# LieFVIN
Code for paper "Lie Group Forced Variational Integrator Networks for Learning and Control of Robot Systems"

Please check out the paper here: https://arxiv.org/pdf/2211.16006.pdf

## Dependencies
The code has been tested with Ubuntu 18.04, Python 3.8 and the following packages:

```torchdiffeq 0.1.1```

```torch 1.11.0```

```gym-pybullet-drones: https://github.com/utiasDSL/gym-pybullet-drones```

```gym 0.21.0```

```numpy 1.23.4```

```scipy 1.9.2```

```matplotlib 3.6.1```

```pyglet 1.5.27```

Install mpc.pytorch:
```git clone git@github.com:locuslab/mpc.pytorch.git && cd mpc.pytorch && pip3 install -e .```

## Demos with pendulum
Run ```python ./examples/pendulum/train_pend_SO3.py``` to train the model with data collected from the pendulum environment. It might take some time to train. A pretrained model is stored in ``` ./examples/pendulum/data/run1/pendulum-so3ham-vin-10p-6000.tar```


Run ```python ./examples/pendulum/analyze_pend_SO3_genfig.py``` to plot the generalized mass inverse M^-1(q), the potential energy V(q), and the control coefficient g(q)

Run ```python ./examples/pendulum/rollout_pend_SO3.py``` to verify that our framework respect energy conservation and SE(3) constraints by construction, and plots phase portrait of a trajectory rolled out from our dynamics.

Run ```python ./examples/pendulum/control_mpc_pend_SO3.py``` to test our MPC controller with the learned dynamics.

## Demos with quadrotor

Run ```python ./examples/quadrotor/train_quad_pyb_SE3_trainingb.py``` to train the model with data collected from the pybullet drone environment. It might take some time to train. A pretrained model is stored in ``` ./examples/quadrotor_pyb/data/run1/quadrotor-se3fvin-vin-5p5-40000.tar```

Run ```python ./examples/quadrotor/analyze_quadrotor_SE3_genfig.py``` to plot the generalized mass inverse M^-1(q), the potential energy V(q), and the control coefficient g(q)

Run ```python ./examples/quadrotor/rollout_quadrotor_SE3.py``` to verify that our framework respect energy conservation and SE(3) constraints by construction, and plots phase portrait of a trajectory rolled out from our dynamics.

Run ```python ./examples/quadrotor/control_mpc_quadrotor_SE3.py``` to test our MPC controller with the learned dynamics.


## Citation
If you find our papers/code useful for your research, please cite our work as follows.

1. V. Duruisseaux, T. Duong, M. Leok, N. Atanasov. [Lie Group Forced Variational Integrator Networks for Learning and Control of Robot Systems](https://arxiv.org/pdf/2211.16006.pdf). arxiv 2022..

 ```bibtex
@article{duruisseaux2022lie,
  title={Lie Group Forced Variational Integrator Networks for Learning and Control of Robot Systems},
  author={Duruisseaux, Valentin and Duong, Thai and Leok, Melvin and Atanasov, Nikolay},
  journal={arXiv preprint arXiv:2211.16006},
  year={2022}
}
```