# Testing autodiff for solving the Poisson equation
The code in this repository is used to test the autodiff package for solving the Poisson equation ($\Delta u = f$) on arbitrary surfaces. We use an MLP to approximate the function $u$, and the loss function is $\|\Delta u_{MLP} - f\|^2$.
The experiments are set up to check if one observes a convergence pattern as the number of degrees of freedom (tunable weights of the MLP) is increased.

## Experiments
We primarily built different classes of experiments, with each one being a little more involved than the previous one, to check where autodiff might fail. 
Our experiments can be summarised in the following table:
| Parameter Domain<br> \ <br> Experiment | 2D Plane | 2D Plane | Sphere |
| ---- | ----- | ----- | ----- |
|   | _Dirichlet_ | _Neumann_ |  |
| On the parameter domain | Convergence | Convergence | Convergence |
| Using exact normals | Convergence | Convergence | Convergence |
| MLP learns exact normals  | Convergence | Convergence | Convergence |
| MLP learns mesh normals | Convergence | Convergence | Convergence|
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
1. __test_exact__ : Solve the Poisson equation on the plane and sphere(experiment 1).
2. __test_exact_parameterization__: Solve the Poisson equation on heightfield and ellipsoid (experiment 2,3,4).
3. __test_computed_parameterization__: Solve the Poisson equation on arbitrary mesh (experiment 5).

## Run
Go to the required directory: __test_exact__, __test_exact_parameterization__ or __test_computed_parameterization__.

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
The _config.json_ in __test_exact__ looks like the following
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

The _config.json_ in __test_exact_parameterization__ looks like the following
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

The _config.json_ in __test_computed_parameterization__ looks like the following
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

## Pretrained models
We test using multiple analytical functions for all experiments. The pretrained models are stored in `poisson_results` for each experiment. The models are trained using the `train` command.
The directory also contains the convergence plots for each example using the `plot` command on the trained models and a visualization of the true function, predicted function after training and the error.

## Stopping Criteria and Runtime
For training all models, we use Adam with a scheduler that lowers the learning rate if the loss doesn't go down after a certain number of iterations. We stop training after 150000 iterations (all networks saturate by that point). For most experiments, it takes about an hour to train a single model.
