# HEFT basis spectra compared with CSSTLaB

The [HEFT notebook](../heft.ipynb) shows 15 spectra at $z=0$ from five advected fields. Each panel compares a single pmwd/JAX realization (blue) with the CSSTLaB 1.2 prediction for the same cosmology (orange). The figure uses the archived production comparison made with the standalone JAX HEFT implementation from which the notebook interface was developed. The quickstart cells use their own resolution and initial conditions.

[View the figure at full resolution](heft_csst_spectra.png) · [Download the plotted spectra](heft_csst_spectra.csv)

## Reading the figure

The five auto-spectra use a logarithmic vertical axis. The ten cross-spectra retain their signs on a symmetric logarithmic axis with a linear region around zero. The shaded interval is $0.15\leq k<0.4\,h\,\mathrm{Mpc}^{-1}$. It contains 11 populated shells. The following values are the unweighted medians of $100(P_{\rm HEFT}/P_{\rm CSST}-1)$ over those shells, one median per auto-spectrum.

| Field | $O_1$ | $O_d$ | $O_{d2}$ | $O_{s2}$ | $O_L$ |
|---|---:|---:|---:|---:|---:|
| Median difference | −0.85% | −0.66% | −1.14% | −1.46% | −1.83% |

The $O_{d2}$–$O_L$ cross-spectrum illustrates why the signs matter. Within the shaded interval the measured spectrum changes sign between $k=0.219$ and $0.240$, and the reference changes sign between $k=0.263$ and $0.289$ (all in $h\,\mathrm{Mpc}^{-1}$). These brackets are adjacent shell centers. The auto-spectrum medians summarize this realization. Individual bins and cross-spectra can differ more.

## Redraw from the published table

Opening the notebook or the PNG requires no simulation or emulator installation. To redraw the figure, change into `docs/examples/heft_validation/` and run the Python block below in an environment with NumPy and Matplotlib. It writes `heft_csst_reproduced.png`. This render used NumPy 2.3.5 and Matplotlib 3.10.8 with the Agg backend.

Each row of the CSV is one operator pair and one shell. `k_lo`, `k_hi` and `k_mean` are in $h\,\mathrm{Mpc}^{-1}$. `Nmodes` counts full-lattice modes including Hermitian partners. `P_heft` and `P_csst` hold the two signed shell averages. The corresponding `valid_*` flag is 1 for a usable value. The table preserves empty shells. It contains the `P_NEW` and `P_CSST` series of the archived `analysis_results.npz` (SHA-256 `4c77c27564cda20537cf3a7debf76a1f4c15d9f797744c7cfd3cf52c0d3934ff`).

<details>
<summary>Python code for the published figure</summary>

```python
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np

table = np.genfromtxt('heft_csst_spectra.csv', delimiter=',', names=True,
                      dtype=None, encoding='utf-8')
fields = ['O1', 'Od', 'Od2', 'Os2', 'OL']
symbols = dict(zip(fields, ['1', r'\delta', r'\delta^2', r's^2', r'\nabla^2\delta']))
fig, axes = plt.subplots(5, 3, sharex=True, figsize=(10.8, 12.8))
fig.subplots_adjust(left=.085, right=.987, bottom=.06, top=.925,
                    hspace=.48, wspace=.31)
for ax, (i, j) in zip(axes.flat, combinations_with_replacement(fields, 2)):
    rows = table[(table['field_i'] == i) & (table['field_j'] == j)]
    k = rows['k_mean']
    valid = (rows['Nmodes'] > 0) & np.isfinite(k)
    measured = np.where(valid & (rows['valid_heft'] == 1), rows['P_heft'], np.nan)
    reference = np.where(valid & (rows['valid_csst'] == 1), rows['P_csst'], np.nan)
    ax.set_xscale('log')
    ax.set_xlim(.011, .86)
    if i == j:
        ax.set_yscale('log')
    else:
        peak = np.nanmax(np.abs([measured, reference]))
        ax.set_yscale('symlog', linthresh=1e-3 * peak)
    ax.axvspan(.15, .4, color='#eef1f4', zorder=-2)
    heft, = ax.plot(k, measured, color='#175080', lw=1.2, marker='o', ms=3.0,
                    markerfacecolor='white', markeredgewidth=.8, label='HEFT measurement')
    csst, = ax.plot(k, reference, color='#bc571d', lw=1.5, ls='--', marker='.',
                    ms=2.4, label='CSSTLaB 1.2')
    ax.set_title(fr'$P_{{{symbols[i]},\,{symbols[j]}}}$', fontsize=13, loc='left', pad=2)
    exponent = 3 - 2 * ((i == 'OL') + (j == 'OL'))
    units = r'$h^{-1}\,\mathrm{Mpc}$' if exponent == 1 else fr'$(h^{{-1}}\,\mathrm{{Mpc}})^{{{exponent}}}$'
    ax.set_ylabel(units, fontsize=9)
    if i != j:
        # Choose sign-preserving major ticks separated in display coordinates.
        lo, hi = ax.get_ylim()
        candidates = [v for v in ax.get_yticks() if lo <= v <= hi and
                      (v == 0 or abs(v) >= 1e-3 * peak)]
        transform = ax.yaxis.get_transform().transform
        min_gap = .13 * (transform(hi) - transform(lo))
        ordered = sorted(candidates, key=lambda v: -abs(v))
        kept = [0.] if lo < 0 < hi else []
        for v in ordered:
            if all(abs(transform(v) - transform(u)) >= min_gap for u in kept):
                kept.append(v)
        if kept:
            ax.set_yticks(sorted(kept))
    ax.tick_params(which='both', direction='in', labelsize=9)
    ax.grid(axis='y', color='#dce1e4', lw=.55)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('HEFT basis spectra at $z=0$', y=.993, fontsize=17)
fig.legend(handles=[heft, csst], loc='upper center', bbox_to_anchor=(.5, .966),
           frameon=False, ncol=2, fontsize=11, handlelength=3.3)
for ax in axes[-1]:
    ax.set_xticks([.02, .05, .1, .2, .5], ['0.02', '0.05', '0.1', '0.2', '0.5'])
    ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$', fontsize=11)
fig.savefig('heft_csst_reproduced.png', dpi=165)
```

</details>

## Reproduce the physical calculation

### 1. Set the simulation and linear input

The measurement uses one periodic realization at $z=0$. These are the production settings for the figure. The runnable example earlier in the notebook uses the pmwd quickstart settings.

| Quantity | Production value |
|---|---|
| Box length and volume | $L=512\,h^{-1}\mathrm{Mpc}$, $V=L^3$ |
| Particles, force mesh, HEFT output | $512^3$, $768^3$, $256^3$, respectively |
| Evolution | Second-order Lagrangian initial conditions at $a=1/32$ ($z=31$), then 62 native pmwd steps to $a=1$, with maximum $\Delta a=1/64$ |
| Precision | float64 cosmological calculations, float32 particle displacements and fields, complex64 native modes |
| Cosmology | $\Omega_b=0.0486$, $\Omega_\mathrm{cdm}=0.2603$, $\Omega_m=\Omega_{cb}=0.3089$, $h=0.6774$, $n_s=0.9667$ |
| Primordial amplitude and dark energy | $A_s=2.142\times10^{-9}$ at $k_*=0.05\,\mathrm{Mpc}^{-1}$, $w_0=-1$, $w_a=0$, $\Omega_k=0$, $\sum m_\nu=0$ |

Compute the **unsmoothed, linear cold-dark-matter plus baryon spectrum at $z=1$** using CLASS. The run used `classy==3.4.2.0` (CLASS v3.4.2) with the CEmulator 2.1.1 CLASS wrapper (commit `480aa9f5ed22de1b94c845ad6baf922922e82b8d`). It used synchronous gauge, $T_\mathrm{CMB}=2.7255\,\mathrm{K}$, $N_\mathrm{ur}=3.046$, $N_\mathrm{ncdm}=0$ and $\Omega_\Lambda=0$, with the flat-universe remainder in the $w=-1$ fluid. With zero massive neutrinos, total matter and $cb$ coincide. If $k$ is in $h\,\mathrm{Mpc}^{-1}$, tabulate $P_h(k,1)=h^3 P_\mathrm{CLASS}(h k,1)$ in $(h^{-1}\mathrm{Mpc})^3$. Use 4,096 logarithmically spaced samples from $10^{-5}$ to $10\,h\,\mathrm{Mpc}^{-1}$. Interpolate $\log P_h$ against $\log k$ with a monotone cubic interpolator (`scipy.interpolate.PchipInterpolator`, extrapolation disabled). The furthest particle-grid Fourier corner is $\sqrt{3}\pi\,h\,\mathrm{Mpc}^{-1}$. As a normalization check, CLASS gives $\sigma_8(z=0)=0.83065447$ with this $A_s$.

The following recreates the table from the [CEmulator CLASS wrapper](https://github.com/czymh/csstemu/blob/480aa9f5ed22de1b94c845ad6baf922922e82b8d/CEmulator/utils.py#L87-L132). This wrapper takes $H_0=67.74$ in $\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}$. The CSSTLaB spectrum interface in step 4 takes $h=0.6774$.

```python
import numpy as np
from CEmulator.utils import useCLASS
from scipy.interpolate import PchipInterpolator

h = 0.6774
cl = useCLASS(
    [0.0486, 0.3089, 100 * h, 0.9667, 2.142, -1., 0., 0.],
    "0.0000, 1.0000", non_linear=None, kmax=10.,
)
k_tab = np.geomspace(1e-5, 10., 4096)
p_tab = np.array([h**3 * cl.pk_lin(h * k, 1.) for k in k_tab])
log_pk = PchipInterpolator(np.log(k_tab), np.log(p_tab), extrapolate=False)
```

For a pmwd realization, let $\omega(\mathbf k)$ be the dimensionless Fourier white noise in pmwd's orthonormal convention. Construct its native, growth-unscaled modes as

$$
m(\mathbf k)=\omega(\mathbf k)\,
\frac{\sqrt{V\,P_h(k,1)}}{D_\mathrm{PM}(a=1/2)},\qquad m(\mathbf 0)=0.
$$

Use pmwd's own growth function in both the denominator and the evolution. Thus the linear field used for the $z=0$ HEFT weights has ensemble power $[D_\mathrm{PM}(1)/D_\mathrm{PM}(1/2)]^2 P_h(k,1)$. Calling CLASS at $z=0$ here changes the prescription. For independent realizations, draw $\omega$ with `pmwd.white_noise`. For the plotted realization the historical run recovered $\omega$ from the archived seed-0 complex64 pmwd modes by dividing each nonzero mode by the real coefficient from `linear_modes(ones, cosmo, conf)` for its original `Cosmology.from_sigma8(..., sigma8=0.8159)` background. The division was done in double precision. The target $\sqrt{V P_h}$ was computed in float32 and the target modes stored as complex64. The old $\sigma_8$ only enters the recovery of the archived realization and the native background setup. The target amplitude is fixed by $A_s$. The archived mode array is not included with the notebook, so reproducing the plotted numbers exactly is simplest with the supplied spectrum table.

For an independent realization the following sets up the native background, the scale factors and the Fourier noise. Build $m$ from the equation above at each nonzero wavevector, using `np.exp(log_pk(np.log(k)))` for $P_h$. The native mode array has shape `(512, 512, 257)`. Evaluating the amplitudes in slabs along the first axis limits memory use.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pmwd import Configuration, Cosmology, boltzmann, growth, white_noise

conf = Configuration(
    ptcl_spacing=1., ptcl_grid_shape=(512,) * 3, mesh_shape=(768,) * 3,
    a_start=1/32, a_stop=1., a_nbody_maxstep=1/64, lpt_order=2,
    float_dtype=jnp.float32, cosmo_dtype=jnp.float64, pmid_dtype=jnp.int16,
)
cosmo = boltzmann(Cosmology.from_sigma8(
    conf, sigma8=0.8159, Omega_m=0.3089, Omega_b=0.0486, h=0.6774, n_s=0.9667,
), conf)
D_anchor = growth(0.5, cosmo, conf)
omega = white_noise(0, conf)
```

Pass $m$ to native pmwd `lpt` and then `nbody` with the configuration above. Keep the association between each final particle and its original lattice site. The historical calculation used [pmwd at 0552bc0](https://github.com/eelregit/pmwd/tree/0552bc0131755367574d8d51ae4fc08d6c29ecf1).

```python
from pmwd import lpt, nbody
from pmwd.pm_util import fftfreq, fftinv

kx, ky, kz = fftfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)
modes_host = np.empty(omega.shape, dtype=np.complex64)
for start in range(0, 512, 8):
    slab = slice(start, start + 8)
    k = np.asarray(jnp.sqrt(kx[slab]**2 + ky**2 + kz**2))
    nonzero = k > 0
    p = np.zeros(k.shape, dtype=np.float64)
    p[nonzero] = np.exp(log_pk(np.log(k[nonzero].astype(np.float64))))
    amplitude = np.sqrt(np.asarray(512.**3 * p, dtype=np.float32))
    weights = np.asarray(omega[slab], dtype=np.complex128)
    modes_host[slab] = weights * amplitude / float(D_anchor)

modes = jnp.asarray(modes_host)
delta_L0 = fftinv(modes, shape=conf.ptcl_grid_shape, norm=conf.ptcl_spacing)
delta_L0 = delta_L0 * growth(1., cosmo, conf)
particles, observables = lpt(modes, cosmo, conf)
final_particles, _ = nbody(particles, observables, cosmo, conf)
```

### 2. Form and advect the five weights

The historical pipeline reconstructed an unsmoothed real $\delta_\mathrm L(z=0)$ by applying pmwd's native inverse Fourier transform to $m$ and multiplying by $D_\mathrm{PM}(1)$. For the operator calculation, take its forward-normalized real FFT. Multiply that linear field by the radial CSST **field-amplitude** window

$$
W_\mathrm{amp}(k)=\exp\!\left[-\frac{R^2}{4}\max(k-k_p,0)^2\right],
\qquad R=5\,h^{-1}\mathrm{Mpc},\quad k_p=1\,h\,\mathrm{Mpc}^{-1}.
$$

Below $k_p$ the window is unity. Its square gives the window on linear **power**. Apply it once to the density used for the Lagrangian weights. The modes used for particle evolution remain unsmoothed. This is the smoothing prescription of [Zhou et al. (2025), §3.2, arXiv:2506.04671](https://arxiv.org/abs/2506.04671).

Denote the smoothed linear density by $d$. The weights are $1$, $d$, $\nabla^2 d$, $d^2-\langle d^2\rangle$, and $s_{ij}s_{ij}-\langle s_{ij}s_{ij}\rangle$, with $s_{ij}(\mathbf k)=(k_i k_j/k^2-\delta_{ij}/3)d(\mathbf k)$ and zero tidal DC. Form the quadratic products on the $800^3$ padded grid. Split the particle-grid Nyquist modes on padding, remove the means of the quadratic products, and merge both Nyquist signs when projecting back to $512^3$. Carry each weight to its particle's final position and deposit by periodic cloud-in-cell (CIC) assignment on $256^3$ tracer-cell centers (offset $1\,h^{-1}\mathrm{Mpc}$ along each axis). Normalize all deposits by $(256/512)^3=1/8$ and subtract unity from the unit-weight field to obtain the matter overdensity $O_1$.

The standalone production implementation was `HEFT_pmwd_JAX_csst.py` (SHA-256 `f6b50af8cefbfc52fb903322b6eeef942f76da49c3f43cb80d14d8da32f330a0`). With the notebook's HEFT interface, the corresponding call for an existing unsmoothed real linear field and the final native particles is

```python
import jax.numpy as jnp
from pmwd.heft import heft_operators

linear_field = jnp.fft.rfftn(jnp.asarray(delta_L0, dtype=jnp.float32), norm="forward")
fields = heft_operators(
    linear_field, final_particles, conf,
    output_grid=256, smoothing="csst", radius=5.0, pivot=1.0,
)
```

### 3. Measure the signed auto- and cross-spectra

Reorder pmwd's output `[O1, Od, OL, Od2, Os2]` by indices `[0, 1, 3, 4, 2]` to get the emulator basis $[1,\delta,\delta^2,s^2,\nabla^2\delta]$. Compute a float64, forward-normalized real FFT of each Eulerian field. For every nonzero mode $\mathbf k=2\pi\mathbf n/L$ and each pair $i\le j$, form

$$
p_{ij}(\mathbf k)=\frac{V\,\mathrm{Re}[\tilde O_i(\mathbf k)\tilde O_j^*(\mathbf k)]}{W_\mathrm{CIC}(\mathbf n)^2},
\qquad W_\mathrm{CIC}(\mathbf n)=\prod_{a=1}^{3}\mathrm{sinc}^2(n_a/256),
$$

where $\mathrm{sinc}(x)=\sin(\pi x)/(\pi x)$. Both measured fields were painted once, giving the squared field-assignment window in the denominator. Preserve the signs of cross-spectra and use no shot-noise subtraction.

Bin by $|\mathbf k|$ with edges `np.r_[10**(-2 + 0.04*np.arange(48)), np.pi/4]` in $h\,\mathrm{Mpc}^{-1}$. Count the upper endpoint of the last bin (half of the output Nyquist frequency). For a real FFT, weight modes of the stored last axis by two except on its zero and Nyquist planes, where the weight is one. Plot the mode-weighted mean $p_{ij}$ against the mode-weighted mean $k$ in each nonempty shell. A count check gives 41 nonempty bins out of 48 and 1,097,916 full-lattice modes in total. The units of each spectrum are $(h^{-1}\mathrm{Mpc})^{3-2n_L}$, where $n_L$ counts its Laplacian factors.

The following also builds the list of distinct mode magnitudes for the reference query.

```python
n, box = 256, 512.
edges = np.r_[10**(-2 + 0.04*np.arange(48)), np.pi/4]
nx = np.rint(np.fft.fftfreq(n) * n).astype(int)[:, None, None]
ny = nx.reshape(1, n, 1)
nz = np.arange(n//2 + 1)[None, None, :]
k_grid = (2*np.pi/box) * np.sqrt(nx**2 + ny**2 + nz**2)
use = (k_grid > edges[0]) & (k_grid <= edges[-1])
k_mode = k_grid[use]
shell = np.searchsorted(edges, k_mode, side='left') - 1
mult = np.broadcast_to(np.where((nz == 0) | (nz == n//2), 1, 2), use.shape)[use]
count = np.bincount(shell, weights=mult, minlength=len(edges)-1)
unique_k, inverse = np.unique(k_mode, return_inverse=True)

def shell_mean(values):
    total = np.bincount(shell, weights=mult * values, minlength=len(count))
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)

k_mean = shell_mean(k_mode)
w_cic = (np.sinc(nx/n) * np.sinc(ny/n) * np.sinc(nz/n))[use]**2
fourier = []
for index in [0, 1, 3, 4, 2]:
    field = np.asarray(fields[index], dtype=np.float64)
    fourier.append(np.fft.rfftn(field, norm="forward")[use] / w_cic)
pairs = list(zip(*np.triu_indices(5)))
p_heft = np.array([shell_mean(box**3 * (fourier[i] * fourier[j].conj()).real)
                   for i, j in pairs])
```

### 4. Evaluate CSSTLaB on the same modes

Query the [CSST Lagrangian-bias emulator, release source](https://github.com/ShurenZhou1999/csstlab/tree/fedd7a3bb1df45785064e550822b7a369afd0683) at $z=0$. Use the sorted unique magnitudes of the measured Fourier modes within its query range $0.001\le k\le1\,h\,\mathrm{Mpc}^{-1}$, before binning. The parameter order is $(\Omega_b,\Omega_{cb},h,n_s,10^9 A_s,w_0,w_a,\sum m_\nu/\mathrm{eV})$.

```python
import numpy as np
from csstlab import Emulator

model = Emulator(remake=False)
model.set_k_and_z(unique_k, 0.)
params = np.array([0.0486, 0.3089, 0.6774, 0.9667, 2.142, -1., 0., 0.])
matrix = np.asarray(model(params))
mask = np.asarray(model._Emulator__Mask_k, dtype=bool)  # per-pair support
if matrix.ndim == 4:  # drop the singleton redshift axis when present
    matrix = np.squeeze(matrix, axis=0)
if mask.ndim == 4:
    mask = np.squeeze(mask, axis=0)
assert matrix.shape == mask.shape == (6, 6, len(unique_k))
i, j = np.triu_indices(5)
theory = matrix[i, j, :]
supported = mask[i, j, :]
```

Use the supplied trained model files. The production call used the same prediction routine with automatic retraining and the unused emulator-error covariance disabled.

The upper triangle of the first five fields yields five auto-spectra and ten cross-spectra. The sixth emulator field is $\delta^3$. Expand each prediction back to its individual Fourier modes and average over exactly the same shell membership and multiplicities as the measurement. Retain an emulator point only if every represented mode in that shell passes the pair's support mask. Applying the emulator at the plotted mean $k$ would give a different average. Compare the resulting emulator spectra directly with the deconvolved measurements.


```python
p_csst = np.full_like(p_heft, np.nan)
for row in range(len(pairs)):
    usable = supported[row, inverse]
    n_good = np.bincount(shell, weights=mult * usable, minlength=len(count))
    good_bin = (count > 0) & (n_good == count)
    average = shell_mean(np.where(usable, theory[row, inverse], 0.))
    p_csst[row, good_bin] = average[good_bin]
```
