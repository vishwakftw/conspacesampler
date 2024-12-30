### How to use these scripts?

First, install the package `conspacesampler` using `pip install .` from the topmost directory containing `setup.py`.

Semantics of individual scripts are detailed below.

##### Script 1

```bash
python Dirichlet_MAPLA_Dikin_comp.py \
    --dimension <d> \
    --alpha_min <amin> \
    --alpha_max <amax> \
    --num_iters <K> \
    --stepsize_mapla <hmapla> \
    --stepsize_dikin <hdikin> \
    --run_index <i> \
    --num_particles <N> \
    --progress_file ./progress_dirichlet_mapla_dikin.txt \
    --loss_progress_file ./loss_progress_dirichlet_mapla_dikin.txt
```
This will create two CSV files in the current directory named

- `progress_dirichlet_mapla_dikin.txt`, and
    - Each line of `progress_dirichlet_mapla_dikin.txt` will consist of 11 entries.
        1. `alg`: one of `MAPLA` or `Dikin`,
        2. `dim`: `<d>`,
        3. `amin`: `<amin>`,
        4. `amax`: `<amax>`,
        5. `num_iters`: `<K>`,
        6. `stepsize`: `<hmapla>` if `alg == MAPLA`, `<hdikin>` if `alg == Dikin`,
        7. `run_index`: `<i>`,
        8. `num_particles`: `<N>`,
        9. `total_time`: time taken to complete `<K>` iterations of `alg` with the above settings,
        10. `avg_acceptance`: average acceptance rate computed as the average proportion of particles accepted after `<K> / 10` iterations,
        11. `loss`: the empirical 2-Wasserstein distance between the ground truth and particles at the final iteration.

- `loss_progress_dirichlet_mapla_dikin.txt`.
    - Each line of `loss_progress_dirichlet_mapla_dikin.txt` will consist of 9 entries.
        1. `alg`: one of `MAPLA` or `Dikin`,
        2. `dim`: `<d>`,
        3. `amin`: `<amin>`,
        4. `amax`: `<amax>`,
        5. `itr`: current iteration number,
        6. `stepsize`: `<hmapla>` if `alg == MAPLA`, `<hdikin>` if `alg == Dikin`,
        7. `run_index`: `<i>`,
        8. `num_particles`: `<N>`,
        9. `w2_val`: the empirical 2-Wasserstein distance between the ground truth and particles at the current iteration.
        10. `ed_val`: the energy distance between the ground truth and particles at the current iteration.

##### Script 2

```bash
python Dirichlet_MAPLA_Dikin_scaling_h.py \
    --algorithm <alg> \
    --dimension <d> \
    --num_iters <K> \
    --stepsize <h> \
    --run_index <i> \
    --num_particles <N> \
    --progress_file ./progress_dirichlet.txt
```

The semantics are similar to `Dirichlet_MAPLA_Dikin_comp.py`, but here we pass fewer options.
This will create a single CSV file named `progress_dirichlet.txt`, and each row of this file will consist of 9 entries.
These are the same of those in each line of `progress_dirichlet_mapla_dikin.txt` except the `amin` and `amax` entries.

##### Script 3

```bash
python BayesianLogReg_MAPLA_Dikin_comp.py \
    --dimension <dim> \
    --num_iters <K> \
    --stepsize_scale <Ch> \
    --domain_index <di> \
    --run_index <i> \
    --num_particles <N> \
    --progress_file ./progress_bayeslogreg_mapla_dikin.txt \
    --error_progress_file ./error_progress_bayeslogreg_mapla_dikin.txt \
    --error_quantile_file ./quantile_progress_bayeslogreg_mapla_dikin.txt
```

This will create 3 CSV files in the current directory, which are
- `progress_bayeslogreg_mapla_dikin.txt`
    - Each line of `progress_bayeslogreg_mapla_dikin.txt` will consist of 11 entries.
        1. `alg`: one of `MAPLA` or `Dikin`,
        2. `dim`: `<d>`,
        3. `num_iters`: `<K>`,
        4. `stepsize_scale`: `<Ch>`,
        5. `domain_index`: `<di>`,
        6. `run_index`: `<i>`,
        7. `num_particles`: `<N>`,
        8. `total_time`: time taken to complete `<K>` iterations of `alg` with the above settings,
        9. `avg_acceptance`: average acceptance rate computed as the average proportion of particles accepted after `<K> / 10` iterations,
        10. `<error>`: L1 distance of the sample mean of the particles after the final iteration to the true parameter normalised by dimension.
        11. `<nll>`: the negative log-likelihood of the data computed w.r.t. the sample mean of the particles after the final iteration.

- `error_progress_bayeslogreg_mapla_dikin.txt`
    - Each line of `error_progress_bayeslogreg_mapla_dikin.txt` will consist of 9 entries.
        1. `alg`: one of `MAPLA` or `Dikin`,
        2. `dim`: `<d>`,
        3. `stepsize_scale`: `<Ch>`,
        4. `domain_index`: `<di>`,
        5. `run_index`: `<i>`,
        6. `num_particles`: `<N>`,
        7. `itr`: current iteration number,
        8. `<error>`: L1 distance of the sample mean of the particles at the current iteration to the true parameter normalised by dimension.
        9. `<nll>`: the negative log-likelihood of the data computed w.r.t. the sample mean of the particles at the current iteration.

- `quantile_progress_bayeslogreg_mapla_dikin.txt`
    - Each line of `quantile_progress_bayeslogreg_mapla_dikin.txt` will consist of 10 entries.
        1. `alg`: one of `MAPLA` or `Dikin`,
        2. `dim`: `<d>`,
        3. `stepsize_scale`: `<Ch>`,
        4. `domain_index`: `<di>`,
        5. `run_index`: `<i>`,
        6. `num_particles`: `<N>`,
        7. `itr`: current iteration number,
        8. `q25`: first quartile (or 0.25-quantile) of the absolute error (normalised by dimension) between the `<N>` particles to the ground truth,
        9. `q50`: second quartile (or 0.5-quantile) of the absolute error (normalised by dimension) between the `<N>` particles to the ground truth,
        10. `q75`: third quartile (or 0.75-quantile) of the absolute error (normalised by dimension) between the `<N>` particles to the ground truth.

Passing "NA" or not passing arguments to either `--error_progress_file` or `--error_quantile_file` will result in the error or quantile files respectively not being created.

---

These are the relevant scripts to reproduce the figures in the paper: [High-accuracy sampling from constrained spaces with the Metropolis-adjusted Preconditioned Langevin Algorithm](https://arxiv.org/abs/2412.18701).

##### Citation

```
@misc{srinivasan2024high,
      title={High-accuracy sampling from constrained spaces with the Metropolis-adjusted Preconditioned Langevin Algorithm}, 
      author={Vishwak Srinivasan and Andre Wibisono and Ashia Wilson},
      year={2024},
      eprint={2412.18701},
      archivePrefix={arXiv},
      primaryClass={stat.CO},
      url={https://arxiv.org/abs/2412.18701}, 
}
```