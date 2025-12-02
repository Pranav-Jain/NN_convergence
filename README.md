# Testing autodiff for solving the Poisson equation
The code in this repository is used to test the autodiff package for solving the Poisson equation ($\Delta u = f$) on arbitrary surfaces. We use an MLP to approximate the function $u$, and the loss function is $\|\Delta u_{MLP} - f\|^2$.
The experiments are set up to check if one observes a convergence pattern as the number of degrees of freedom (tunable weights of the MLP) is increased.

## Experiments
We primarily built different classes of experiments, with each one being a little more involved than the previous one, to check where autodiff might fail. 
Our experiments can be summarised in the following table:
| Parameter Domain<br> \ <br> Experiment | 2D Plane | 2D Plane | Sphere |
| ---- | ----- | ----- | ----- |
|   | _Dirichlet_ | _Neumann_ |  |
| On the parameter domain | [13/15](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact/poisson_results/2d/dirichlet/domain_-1.0to1.0/convergence_summary.json) | [10/15](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact/poisson_results/2d/neumann/domain_-1.0to1.0/convergence_summary.json) | [4/4](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact/poisson_results/3d/dirichlet/domain_-1.0to1.0/convergence_summary.json) |
| Using exact normals | [14/15](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact_parameterization/poisson_results/heightfield/noNN/dirichlet/domain_-1.0to1.0/convergence_summary.json) | [11/12](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact_parameterization/poisson_results/heightfield/noNN/neumann/domain_-1.0to1.0/convergence_summary.json) | [9/12](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact_parameterization/poisson_results/ellipsoid/noNN/dirichlet/domain_-1.0to1.0/convergence_summary.json) |
| MLP learns exact normals  | [11/12](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact_parameterization/poisson_results/heightfield/withNN/dirichlet/domain_-1.0to1.0/convergence_summary.json) | [6/12](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact_parameterization/poisson_results/heightfield/withNN/neumann/domain_-1.0to1.0/convergence_summary.json) | [8/12](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact_parameterization/poisson_results/ellipsoid/withNN/dirichlet/domain_-1.0to1.0/convergence_summary.json) |
| MLP learns mesh normals | [9/12](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact_parameterization/poisson_results/heightfield/withNN_mesh/dirichlet/domain_-1.0to1.0/convergence_summary.json) | [4/12](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact_parameterization/poisson_results/heightfield/withNN_mesh/neumann/domain_-1.0to1.0/convergence_summary.json) | [9/12](https://github.com/Pranav-Jain/NN_convergence/blob/main/test_exact_parameterization/poisson_results/ellipsoid/withNN_mesh/dirichlet/domain_-1.0to1.0/convergence_summary.json) |
| MLP learns normals<br> on arbitrary mesh | - | - | - |

1. The first experiment is to solve the Poisson equation directly on a plane with Dirichlet and Neumann boundary and on the surface of the sphere
2. The second experiment is to solve on the heightfield (2d plane as parameter domiain) and ellipsoid (sphere as parameter domain) with exact normals
3. Same as the second experiment, but instead, an MLP is used to learn the exact normals
4. Same as the third experiment, but instead, an MLP is used to learn the mesh normals
5. Same as the fourth experiment, but instead, an MLP is used to learn the mesh normals of an arbitrary mesh

## Setup
This code is tested on _x64 linux_ platform using _Python 3.12.4_.
The code should run out of the box if all the required packages are installed.

The repository has three directories
1. [test_exact](https://github.com/Pranav-Jain/NN_convergence/tree/main/test_exact) : Solve the Poisson equation on the plane and sphere(experiment 1).
2. [test_exact_parameterization](https://github.com/Pranav-Jain/NN_convergence/tree/main/test_exact_parameterization): Solve the Poisson equation on heightfield and ellipsoid (experiment 2,3,4).
3. [test_computed_parameterization](https://github.com/Pranav-Jain/NN_convergence/tree/main/test_computed_parameterization): Solve the Poisson equation on arbitrary mesh (experiment 5).

## Run
Go to the required directory: [test_exact](https://github.com/Pranav-Jain/NN_convergence/tree/main/test_exact), [test_exact_parameterization](https://github.com/Pranav-Jain/NN_convergence/tree/main/test_exact_parameterization) or [test_computed_parameterization](https://github.com/Pranav-Jain/NN_convergence/tree/main/test_computed_parameterization).

Each directory has a file _config.json_.

After setting the _config.json_, run the following command:
```sh
python3 test_convergence.py [1|2|....]
```
The last argument is the example number you want to run for.
Use the `"operation"` argument to switch between training and inference.

### Config Files
The config file for each directory look like the following:
#### <u>test_exact</u>
The _config.json_ in [test_exact](https://github.com/Pranav-Jain/NN_convergence/tree/main/test_exact) looks like the following
```sh
{
    "dimension": [2 | 3],
    "bc": ["dirichlet", "neumann"],
    "operation": ["train", "plot"],
    "threshold": 0.0,
    
    "domain" : {
        "min": -1.0,
        "max": 1.0
    },

    "architecture" : {
        "lr" : 1e-3,
        "num_layers" : 3,
        "grad_clip" : 10.0,
        "scheduler_patience" : 1000,
        "num_samples" : 10000,
        "max_iter" : 150000
    }
}
``` 

#### <u>test_exact_parameterization</u>
In the code, we choose 0.5*(x^2 + y^2) as the heightfield and x^2/9 + y^2/4 + z^2 = 1 as the ellipsoid. These are the two surfaces we solve the Poisson equation on.

The _config.json_ in [test_exact_parameterization](https://github.com/Pranav-Jain/NN_convergence/tree/main/test_exact_parameterization) looks like the following
```sh
{
    "surface": ["heightfield", "ellipsoid"],
    "bc": ["dirichlet", "neumann"],
    "NN": ["noNN", "withNN", "withNN_mesh"],
    "operation": ["train", "plot"],
    "threshold": 0.0,

    "domain" : {
        "min": -1.0,
        "max": 1.0
    },

    "architecture" : {
        "lr" : 1e-3,
        "num_layers" : 3,
        "grad_clip" : 10.0,
        "scheduler_patience" : 1000,
        "num_samples" : 10000,
        "max_iter" : 150000
    }
}
``` 

#### <u>test_computed_parameterization</u>
In the code, we choose an arbitrary mesh as the domain.

The _config.json_ in [test_computed_parameterization](https://github.com/Pranav-Jain/NN_convergence/tree/main/test_computed_parameterization) looks like the following
```sh
{
    "surface": ["handclosed", "bunny"],
    "bc": ["dirichlet", "neumann"],
    "operation": ["train", "plot"],
    "threshold": 0.0,

    "domain" : {
        "min": -1.0,
        "max": 1.0
    },

    "architecture" : {
        "lr" : 1e-3,
        "num_layers" : 3,
        "grad_clip" : 10.0,
        "scheduler_patience" : 1000,
        "num_samples" : 10000,
        "max_iter" : 150000
    }
}
```

## Checking Convergence
The script [check_convergence.py](https://github.com/Pranav-Jain/NN_convergence/blob/main/check_convergence.py) runs the inference for trained models for all examples and uses certain heuristics to check for convergence. The current heuristics used are that if for an example, the slope < -0.3 and correlatio_coefficient < -0.5, then we claim convergence.

Use the following command to run the script
```sh
python3 check_convergence.py [test_exact|test_exact_parameterization|test_computed_parameterization]
```
__Note__: Make sure that the operation in the config file for the directory you are running for is set to _"plot"_.

## Pretrained models
We test using multiple analytical functions for all experiments. The pretrained models are stored in __poisson_results__ for each experiment. The models are trained using the `operation="train"` in the config file.
The directory also contains the convergence plots for each example using `operation="plot"` on the trained models and a visualization of the true function, predicted function after training and the error.

## Stopping Criteria and Runtime
For training all models, we use Adam with a scheduler that lowers the learning rate if the loss doesn't go down after a certain number of iterations. We stop training after 150000 iterations (all networks saturate by that point). For most experiments, it takes about an hour to train a single model.
