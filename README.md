# PINNsur: Physics-Informed Neural Networks for PDEs on Curved Surfaces

Official implementation and experimental code for:

> **PINNsur: Physics-Informed Neural Networks for PDEs on Curved Surfaces**

PINNsur is a framework for solving partial differential equations (PDEs) on curved surfaces using physics-informed neural networks (PINNs). The method represents the PDE solution using a coordinate-based neural network and constructs surface differential operators by projecting Euclidean differential operators onto the surface tangent space using the surface normal field.

This repository contains the code used for the experiments in the paper, including:

- Poisson and Helmholtz PDE solves on flat and curved domains
- Dirichlet and Neumann boundary conditions
- Exact and learned surface normals
- PDE solves on arbitrary triangle meshes
- Empirical convergence experiments under increasing network capacity
- Ablation and baseline experiments
- Scripts for reproducing the figures in the paper
- An extended suite of manufactured-solution experiments used to investigate the limitations of the method

---

## Overview

For a scalar solution field `u`, PINNsur represents the solution using an MLP `u_theta`. The governing PDE is enforced through a physics-informed loss.

For example, for the Poisson equation:

$$
\Delta_{\Omega} u = f
$$

the PDE loss is:

$$
\mathcal{L}_{\mathrm{PDE}} = \left\|
\Delta_{\Omega} u_{\theta} - f
\right\|_2^2
$$

For curved surfaces, the Laplace--Beltrami operator is computed using the surface normal field. Depending on the experiment, the normals are either:

1. known analytically,
2. approximated by an MLP from exact normals, or
3. approximated by an MLP from normals computed on a triangle mesh.

The primary convergence experiments study how the solution error changes as the number of trainable parameters of the PDE network is increased.

---

## Repository Structure

The main experimental directories are:

```text
NN_convergence/
├── data/                           # Meshes and geometry data
├── src/                            # Shared source code
├── scripts/                        # Paper figure/reproduction scripts
├── renderings/images/              # Rendered results
│
├── test_exact/                     # PDEs on plane/sphere
├── test_exact_parameterization/    # Heightfield and ellipsoid experiments
├── test_computed_parameterization/ # Arbitrary triangle meshes
│
├── blender_utils/                  # Rendering utilities
├── check_convergence.py            # Convergence analysis
├── env.yaml                        # Conda environment
└── README.md
```

The three main experiment directories correspond to progressively more general geometric settings.

### `test_exact`

Solves the Poisson equation directly on a plane or sphere, corresponding to Experiment 1 below.

### `test_exact_parameterization`

Contains the heightfield and ellipsoid experiments corresponding to Experiments 2--4:

- exact surface normals,
- an MLP trained on exact normals, and
- an MLP trained on mesh-computed normals.

### `test_computed_parameterization`

Contains experiments on arbitrary triangle meshes, corresponding to Experiment 5.

---

# Experiments

We constructed a sequence of experiments with progressively more involved geometric information. In addition to producing the representative examples shown in the paper, these experiments were used to investigate when neural PDE solving with autodifferentiation succeeds or fails.

The complete experimental suite is summarized below.

| # | Experiment | Heightfield / Plane Dirichlet | Heightfield / Plane Neumann | Ellipsoid / Sphere |
|---|---|---:|---:|---:|
| 1 | PDE on parameter domain | 10/10 | 9/10 | 4/4 |
| 2 | Using exact normals | 10/10 | 6/10 | 8/12 |
| 3 | MLP learns exact normals | 10/10 | 7/10 | 8/12 |
| 4 | MLP learns mesh normals | 8/10 | 6/10 | 9/12 |
| 5 | MLP learns normals on arbitrary mesh | Hand: 10/10 | Hand: 10/10 | Bunny: 10/10, Spot: 9/10 |

Each entry reports the number of individual manufactured-solution experiments that exhibit empirical convergence under network refinement.

The experiments correspond to the following settings:

1. **PDE on the parameter domain.**  
   Solve the Poisson equation directly on a plane with Dirichlet and Neumann boundary conditions and directly on the surface of a sphere.

2. **Exact surface normals.**  
   Solve the Poisson equation on a heightfield (with a 2D plane as its parameter domain) and an ellipsoid (with a sphere as its parameter domain), using exact surface normals.

3. **MLP learns exact normals.**  
   Repeat Experiment 2, but train an MLP to approximate the analytically known surface normals.

4. **MLP learns mesh normals.**  
   Repeat Experiment 3, but train the normal MLP using normals computed from a triangle mesh.

5. **Arbitrary triangle meshes.**  
   Train an MLP to approximate mesh normals and solve the PDE directly on arbitrary mesh surfaces such as the hand, bunny, and Spot models.

---

## Extended Convergence Experiments and Failure Cases

The experiments above include a broader set of manufactured solutions than the representative examples shown in the main paper.

The purpose of this extended suite is not only to demonstrate successful PDE solves, but also to investigate the regimes in which the empirical convergence behavior deteriorates.

In particular, we systematically considered functions with increasing:

- spatial frequency, and
- function range / dynamic range.

The non-convergent cases in the extended experiments are concentrated in the challenging regimes discussed in the **Limitations** section of the paper: high-frequency target functions, high-range target functions, and, for some Neumann problems, geometries for which the required normal field contains or is affected by discontinuities.

### Heightfield with Neumann boundary conditions

For the heightfield with Poisson--Neumann boundary conditions, the extended suite gives:

| Normal representation | Converged experiments |
|---|---:|
| Exact normals | 6/10 |
| MLP learns exact normals | 7/10 |
| MLP learns mesh normals | 6/10 |

Importantly, difficult high-frequency/high-range cases can fail even when the **exact surface normals** are supplied to the PDE solver. These failures therefore cannot in general be attributed solely to errors introduced by the learned normal network.

### Ellipsoid

We observe analogous behavior on the ellipsoid:

| Normal representation | Converged experiments |
|---|---:|
| Exact normals | 8/12 |
| MLP learns exact normals | 8/12 |
| MLP learns mesh normals | 9/12 |

The difficult cases again occur for sufficiently oscillatory or high-range functions. In these regimes, increasing the number of trainable parameters does not necessarily result in a consistent reduction of the approximation error.

These extended experiments are intended to make the limitations of the method explicit. PINNsur is not claimed to converge for every manufactured solution or arbitrary PDE instance.

The empirical convergence results in the paper instead study whether the approximation error decreases under increasing neural-network degrees of freedom for the tested PDE instances.

---

# Empirical Convergence Study

The convergence experiments investigate the effect of increasing the effective number of degrees of freedom of the neural PDE solver.

For each PDE instance, we keep the following fixed:

- PDE
- domain/surface
- boundary condition
- sampling procedure
- network architecture family
- training procedure

We then vary the number of trainable parameters (`#W`) of the PDE network.

The PDE network has depth 3, and its width is varied to change the number of trainable parameters.

The relative solution error is:

$$
e =
\frac{
\left\|u_{\theta} - u_{\mathrm{GT}}\right\|_2
}{
\left\|u_{\mathrm{GT}}\right\|_2
}
$$

We then examine how the relative error `e` changes as the number of trainable parameters (`#W`) increases.

All networks are trained until optimization has saturated before the final error is evaluated. Therefore, the refinement variable in these experiments is **model capacity**, rather than training time or the number of sampled points.

---

## Checking Convergence

The script

```text
check_convergence.py
```

runs inference for the trained models and computes the empirical convergence trend.

Run:

```bash
python3 check_convergence.py test_exact
```

or:

```bash
python3 check_convergence.py test_exact_parameterization
```

or:

```bash
python3 check_convergence.py test_computed_parameterization
```

Make sure that the `operation` field in the corresponding configuration file is set to:

```json
"operation": "plot"
```

before running the convergence analysis.

For an individual experiment, the script fits a line to:

```text
log(relative error) vs. log(# trainable parameters)
```

and computes the slope and correlation coefficient.

The automated heuristic labels an individual experiment as exhibiting convergence when:

```text
slope < -0.3
correlation_coefficient < -0.5
```

These numerical thresholds provide an automated way of identifying a sufficiently consistent decreasing error trend in the experimental results. They should be interpreted as an **empirical analysis heuristic**, rather than a theorem-backed convergence guarantee.

---

# Installation

We recommend creating the provided Conda environment:

```bash
conda env create -f env.yaml
conda activate test_convergence
```

The code has been tested on an x64 Linux platform using Python 3.11.

The code should run after all packages specified in `env.yaml` have been installed.

---

# Running the Main Experiments

Navigate to the experiment directory you want to run:

```bash
cd test_exact
```

or:

```bash
cd test_exact_parameterization
```

or:

```bash
cd test_computed_parameterization
```

Each experiment directory contains a `config.json`.

After setting the desired configuration, run:

```bash
python3 test_convergence.py <example_number>
```

For example:

```bash
python3 test_convergence.py 1
```

The final argument specifies the manufactured-solution example to run.

Use the `"operation"` entry in `config.json` to switch between training and plotting/inference.

---

# Configuration Files

## `test_exact`

The `config.json` for `test_exact` has the following structure:

```json
{
    "dimension": [2, 3],
    "bc": ["dirichlet", "neumann"],
    "operation": ["train", "plot"],
    "threshold": 0.0,

    "domain": {
        "min": -1.0,
        "max": 1.0
    },

    "architecture": {
        "lr": 1e-3,
        "num_layers": 3,
        "grad_clip": 10.0,
        "scheduler_patience": 1000,
        "num_samples": 10000,
        "max_iter": 150000
    }
}
```

The options shown in arrays indicate the available choices rather than literal values to be supplied simultaneously.

---

## `test_exact_parameterization`

For these experiments, the two analytic surfaces are the heightfield and ellipsoid.

### Heightfield

The heightfield is defined by:

$$
z = \frac{1}{2}\left(x^2 + y^2\right)
$$

### Ellipsoid

The ellipsoid is defined by:

$$
\frac{x^2}{9}
+
\frac{y^2}{4}
+
z^2
=
1
$$

The configuration has the following structure:

```json
{
    "surface": ["heightfield", "ellipsoid"],
    "bc": ["dirichlet", "neumann"],
    "NN": ["noNN", "withNN", "withNN_mesh"],
    "operation": ["train", "plot"],
    "threshold": 0.0,

    "domain": {
        "min": -1.0,
        "max": 1.0
    },

    "architecture": {
        "lr": 1e-3,
        "num_layers": 3,
        "grad_clip": 10.0,
        "scheduler_patience": 1000,
        "num_samples": 10000,
        "max_iter": 150000
    }
}
```

The normal options correspond to:

```text
noNN        -> use exact normals directly
withNN      -> MLP learns exact normals
withNN_mesh -> MLP learns mesh-computed normals
```

---

## `test_computed_parameterization`

For these experiments, the domain is an arbitrary triangle mesh.

The configuration has the following structure:

```json
{
    "pde": "helmholtz",
    "k": 2.0,
    "surface": ["hand", "bunny"],
    "bc": ["dirichlet", "neumann"],
    "operation": ["train", "plot"],
    "threshold": 0.0,

    "architecture": {
        "lr": 1e-3,
        "num_layers": 3,
        "grad_clip": 10.0,
        "scheduler_patience": 1000,
        "num_samples": 10000,
        "max_iter": 150000
    }
}
```

---

# Training the Surface-Normal Networks

PINNsur can use a neural field to approximate the surface normal field.

For the analytic heightfield and ellipsoid experiments, use:

```text
train_hf_normal.py
train_ellipsoid_normal.py
```

Run:

```bash
python3 train_hf_normal.py exact
```

or:

```bash
python3 train_hf_normal.py mesh
```

and similarly:

```bash
python3 train_ellipsoid_normal.py exact
```

or:

```bash
python3 train_ellipsoid_normal.py mesh
```

The argument specifies whether the normal network should learn:

```text
exact -> analytically computed normals
mesh  -> normals computed from the mesh
```

For arbitrary triangle meshes, use:

```bash
python3 train_NN_mesh_normals.py <meshname>
```

where:

```text
<meshname>.obj
```

is present in the `data/` directory.

---

# Reproducing Paper Figures

All paper figure pipelines are located in:

```text
scripts/figure_scripts/
```

First activate the environment:

```bash
conda activate test_convergence
```

If the repository was freshly cloned, the scripts can be made executable with:

```bash
chmod +x scripts/figure_scripts/run_*.sh
```

To generate all paper figure pipelines:

```bash
for f in scripts/figure_scripts/run_*.sh; do
    echo "Running $f"
    bash "$f"
done
```

Generated outputs are written to:

```text
scripts/images/
scripts/plots/
```

where:

- `scripts/images/` contains rendered figure images, and
- `scripts/plots/` contains convergence and comparison plots.

To run a single figure pipeline, for example:

```bash
bash scripts/figure_scripts/run_moai.sh
```

The main orchestrator used by the figure scripts is:

```text
scripts/run_mesh_pipeline.py
```

### Regenerating convergence summaries

After plotting the trained models, convergence summaries can be regenerated with:

```bash
python3 check_convergence.py test_exact
python3 check_convergence.py test_exact_parameterization
python3 check_convergence.py test_computed_parameterization
```

> **Note:** Training can be computationally expensive. Several paper figure scripts use `--render-only` and assume that pretrained outputs are already available in the corresponding `test_*/*_results` directories.

---

# Pretrained Models and Results

We test multiple analytical/manufactured functions for each experimental setting.

Pretrained models are stored in the corresponding result directories, including `poisson_results` where applicable.

Models are generated by setting:

```json
"operation": "train"
```

in the corresponding configuration file.

The result directories also contain outputs generated from the trained models, including:

- convergence plots,
- visualizations of the ground-truth solution,
- predicted solutions, and
- solution errors.

Set:

```json
"operation": "plot"
```

to perform inference and generate the corresponding plots from trained models.

---

# Training and Stopping Criteria

All models are optimized using Adam.

A learning-rate scheduler reduces the learning rate when the training loss stops improving for a specified number of iterations.

The default PDE-network configuration uses:

```text
Initial learning rate : 1e-3
Network depth         : 3
Training samples      : 10000
Maximum iterations    : 150000
Gradient clipping     : 10.0
```

We train each PDE network for a maximum of **150,000 iterations** to ensure that optimization has saturated.

Although many networks saturate substantially earlier, using the same maximum training budget across network sizes prevents differences in training duration from being confused with the effect of increasing network capacity.

For most experiments, training a single model takes approximately one hour on the hardware used during development.

---

# Scope and Limitations

The convergence experiments in this repository constitute an **empirical study of convergence with respect to increasing neural-network degrees of freedom**.

They should not be interpreted as a theoretical guarantee that PINNsur converges for arbitrary PDEs, geometries, or target functions.

The extended experiments expose several important failure regimes.

### 1. High-frequency solutions

Sufficiently oscillatory target functions can become difficult for the finite neural representation and optimization procedure to resolve.

### 2. High-range solutions

Solutions with a sufficiently large dynamic range can fail to exhibit decreasing approximation error under network refinement.

### 3. Discontinuous normal fields

Neumann boundary conditions can become particularly difficult when the required normal field is discontinuous, for example near sharp corners.

These cases are discussed explicitly in the **Limitations** section and supplemental experiments of the paper.

The fact that some high-frequency and high-range experiments fail even when exact normals are provided also indicates that these failures cannot generally be explained only by inaccuracies in the learned normal representation.

---

# Interpretation of the Results

The convergence experiments should be interpreted at the level of the **individual PDE instance**.

For each instance, we ask whether increasing the number of trainable PDE-network parameters results in a systematic reduction of the relative solution error after optimization has saturated.

The larger experimental suite was deliberately used to probe difficult regimes and therefore contains both convergent and non-convergent examples.

Consequently:

- the results do **not** constitute a theoretical convergence guarantee;
- the method is **not** claimed to converge for every function in the extended suite;
- the observed non-convergent cases help characterize the limitations of PINNsur, particularly for high-frequency, high-range, and discontinuous-normal settings.

---

# Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{pinnsur2026,
    title     = {PINNsur: Physics-Informed Neural Networks for PDEs on Curved Surfaces},
    author    = {Anonymous},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2026}
}
```

> The citation above uses the anonymous submission information. We will update the author list and publication information after the review process.
