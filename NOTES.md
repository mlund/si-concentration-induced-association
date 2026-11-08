# Notes on reproducing Li et al. 2016

## Misprints in the article

Both are corrected in the code.

- **Eq. 1** prints `(2A + 2d + r)`. The site-overlap geometry gives `(2A − 2d + r)`,
  as printed in eq. 14 of Wertheim, J. Chem. Phys. 1986, 85, 2929, and a Monte Carlo
  test in `tests/` confirms it. The printed form is about 11 times too large at contact.
- **Eq. 4** prints `−ρ/X` under the root. Mass action, ρ = ρ0 + Xρ0² (eq. 28 of the same
  paper), requires `+ρ/X`.

The published curves agree with the corrected eq. 1. Peak S(0) for hard spheres with
a patch, read from the top panel of Fig. 5:

| βε | Corrected eq. 1 | Printed eq. 1 | Article |
|---|---|---|---|
| 15 | 1.56 | 1.78 | ≈1.55 |
| 14 | 1.45 | 1.69 | ≈1.44 |
| 12.5 | 1.24 | 1.54 | ≈1.23 |
| 10.5 | 1.01 | 1.28 | ≈1.0 |

## Inputs the article does not state

- Bjerrum length 0.7 nm, and Debye length 0.304/√c nm, for water at 25 °C.
- The other potential parameters are the article's: D = 6 nm, Z_eff = 2, κ_a = 20/D
  and K = 1 kT.

## Digitization

All figures in the article's PDF are raster images. Axes were calibrated from tick
positions at the native 300 ppi. Markers were located by template matching or
colour masks, and lines by colour, in Figs. 2–5. The 14 points in Fig. 5 appear in
two panels and agree within 0.002. These values are estimates and should be replaced
by the original data if those become available.

## Results

**Reproduced**

- Fig. 5, top panel: the patch-only curves and the hard-sphere limit.
- Fig. 5, middle panel, 200 mM: S(0) = 0.871 against 0.868 in the article at φ ≈ 0.022.
- Fig. 3: Baxter lines recomputed from the digitized τ of Fig. 4 agree with all 13
  published lines within 0.005 at q = 0.00498 Å⁻¹.

**Not reproduced**

- Fig. 5, 35 mM, βε = 11.2. Peak S(0) is about 1.16 at φ ≈ 0.016, against 1.236 at
  φ ≈ 0.022 in the article. Monomer fraction at φ ≈ 0.022 is 0.64 against 0.72.
  The cause is unknown. Replacing g00 by the cavity function in the bond term
  reproduces the published monomer fraction but lowers S(0) further, and contradicts
  Wertheim's closure. A scan of βε from 8 to 11.2 and K from 0 to 2.5 found no
  parameters that fit both curves. The choice of closure does not explain it: peak S(0)
  is 1.154 on the same coarse φ grid with both. Nor does the route to S(0) (see below).

**Added comparison, not in the article**

- B2 from the model against the Fig. 2 inset: −1900 against −333 ± 52 nm³ at 35 mM,
  and +363 against +22 ± 28 nm³ at 200 mM. The article's own 35 mM curve starts at the
  same S(0) ≈ 1.03 at φ = 0.001, so this appears to be a property of the model with
  D = 6 nm and not of this implementation.
- S(0) of the Baxter model from the digitized τ against measured S(0): within 0.028
  except at φ = 0.057, 35 mM, where it is 0.057 higher.

## Closures

Eq. 6 of the article matches eq. 15 of Wertheim, J. Stat. Phys. 1984, 35, 35. The
reference-HNC closure is eqs. 33–40 of that paper, orientationally averaged as in its
eqs. 57–58. The article does not name its reference system. Here it is the same
isotropic potential without the patch, solved with the Percus–Yevick closure.

The article reports the two closures as indistinguishable for φ < 0.15. Here S(0)
differs by less than 0.001 at φ ≤ 0.03 in the 35 mM model, and by at most 0.024 across
all cases, at φ = 0.1 to 0.15 (`results/closure_comparison.csv`). The peak of S(0)
changes by at most 0.002.

## Analytic benchmark

Wertheim's 1986 paper solves the Percus–Yevick-like closure exactly in the sticky limit
(eqs. 90 and 96). Extrapolating the finite-range solver to zero range reproduces its
monomer fraction, and its d(βp)/dρ to about 1 part in 10⁴ when computed from the
compressibility relation (eq. 18 of Wertheim 1984). This uses c(r) and ρ0(ρ) alone.

The structure route, S(0) from Ornstein–Zernike, gives a slightly different value in
that limit: 1/S(0) is 0.6–3% above d(βp)/dρ at φ = 0.1–0.3. The two routes need not
agree in an approximate theory. For the article's states they agree to three decimals
at φ ≤ 0.075, including the 35 mM model, so this does not affect Fig. 5. The figures
here use the structure route, as the article describes.

## Not attempted

- Refitting τ, which needs the measured S(q).
