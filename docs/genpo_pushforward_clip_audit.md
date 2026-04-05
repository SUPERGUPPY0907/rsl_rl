# GenPO Pushforward Clip Audit

This note documents the mathematical assumptions behind `GenPOU0Clip` and `GenPOPFClip` and clarifies how they differ from the original GenPO paper and the current `GenPO` implementation in this repository.

## 1. Constant-Jacobian Structure of the Implemented Flow

For one flow step over the doubled dummy action \((x, y) \in \mathbb{R}^{2d}\), the implementation applies

\[
\tilde{x}=x+\Delta t\,v_\theta(y, t), \qquad
\tilde{y}=y+\Delta t\,v_\theta(\tilde{x}, t),
\]

followed by the sequential mixing

\[
x^+ = p \tilde{x} + (1-p)\tilde{y}, \qquad
y^+ = p \tilde{y} + (1-p)x^+.
\]

The coupling part is triangular, hence volume preserving:

\[
\det J_{C_1} = 1, \qquad \det J_{C_2} = 1.
\]

The mixing part is also triangular when read in update order:

\[
J_{M_1}=
\begin{bmatrix}
pI_d & (1-p)I_d \\
0 & I_d
\end{bmatrix},
\qquad
J_{M_2}=
\begin{bmatrix}
I_d & 0 \\
(1-p)I_d & pI_d
\end{bmatrix},
\]

so

\[
\det J_{M_1}=p^d, \qquad \det J_{M_2}=p^d.
\]

Therefore one full step has

\[
\det J_{\text{step}} = p^{2d}.
\]

With \(K\) fixed steps, the implemented flow \(F_{\theta,s}\) satisfies

\[
\det J_{F_{\theta,s}} = p^{2dK},
\]

which is independent of both \(z\) and \(\theta\) as long as \(p\) is a fixed hyperparameter.

## 2. Exact Latent-Energy Log Ratios

Let the base latent be \(z \sim \mathcal{N}(0, I_{2d})\) and \(\tilde{a}=F_{\theta,s}(z)\). Since the Jacobian determinant is a sample-independent constant,

\[
\log \tilde{\pi}_\theta(\tilde{a}\mid s)
=
-\frac{1}{2}\|z_\theta\|^2
- d \log(2\pi)
- 2dK \log p,
\]

where \(z_\theta = F_{\theta,s}^{-1}(\tilde{a})\).

For two policies \(\theta\) and \(\theta_{\text{old}}\), the constant terms cancel:

\[
\ell_{\mathrm{dum}}(s,x,y)
=
\log \frac{\tilde{\pi}_\theta(x,y\mid s)}{\tilde{\pi}_{\theta_{\text{old}}}(x,y\mid s)}
=
\frac{1}{2}\|z_{\text{old}}\|^2 - \frac{1}{2}\|z_\theta\|^2.
\]

This is exactly the quantity used by the new pushforward-clip algorithms.

## 3. Dummy Entropy Is Constant for the Implemented Flow

For a change of variables with constant Jacobian determinant,

\[
H(\tilde{\pi}_\theta(\cdot\mid s))
=
H(\mathcal{N}(0, I_{2d})) + \log |\det J_{F_{\theta,s}}|.
\]

Hence

\[
H(\tilde{\pi}_\theta(\cdot\mid s))
=
H(\mathcal{N}(0, I_{2d})) + 2dK \log p.
\]

When \(p\) is fixed, this entropy does not depend on \(\theta\), so its policy gradient is zero.

Practical implication:

- The GenPO paper writes dummy entropy regularization at the density level.
- The current repository logs an `entropy` metric for `GenPO`, but does not add it into the loss.
- `GenPOU0Clip` and `GenPOPFClip` keep entropy out of the objective and treat it as structurally constant under the fixed-\(p\) assumption.

## 4. Pushforward/Fiber Factorization

Introduce the coordinates

\[
a=\frac{x+y}{2}, \qquad u=\frac{x-y}{2},
\]

so that \(x=a+u\) and \(y=a-u\).

Let \(\bar{\pi}_\theta(a,u\mid s)\) denote the density in \((a,u)\)-coordinates. Define the pushforward policy

\[
\pi_\theta^g(a\mid s)=\int \bar{\pi}_\theta(a,u\mid s)\,du,
\]

and the fiber conditional

\[
q_\theta(u\mid a,s)=\frac{\bar{\pi}_\theta(a,u\mid s)}{\pi_\theta^g(a\mid s)}.
\]

Then

\[
\bar{\pi}_\theta(a,u\mid s)=\pi_\theta^g(a\mid s)\,q_\theta(u\mid a,s),
\]

which yields

\[
r_{\mathrm{dum}}(s,a,u)
=
\frac{\bar{\pi}_\theta(a,u\mid s)}{\bar{\pi}_{\theta_{\text{old}}}(a,u\mid s)}
=
r_{\mathrm{act}}(s,a)\,r_{\mathrm{fib}}(s,a,u),
\]

with

\[
r_{\mathrm{act}}(s,a)
=
\frac{\pi_\theta^g(a\mid s)}{\pi_{\theta_{\text{old}}}^g(a\mid s)},
\qquad
r_{\mathrm{fib}}(s,a,u)
=
\frac{q_\theta(u\mid a,s)}{q_{\theta_{\text{old}}}(u\mid a,s)}.
\]

Taking conditional expectation under \(u \sim q_{\theta_{\text{old}}}(\cdot\mid a,s)\),

\[
r_{\mathrm{act}}(s,a)
=
\mathbb{E}\left[r_{\mathrm{dum}}(s,a,u)\mid s,a\right].
\]

This identity explains why dummy-space clipping is more conservative than pushforward clipping: clipping acts before the fiber average.

## 5. Section Estimator and Its Bias-Variance Tradeoff

The diagonal section sets \(u=0\), equivalently \(x=y=a\). Its log ratio is

\[
\ell_0(s,a)
=
\frac{1}{2}\|z_{\text{old}}(s,a,a)\|^2
- \frac{1}{2}\|z_\theta(s,a,a)\|^2.
\]

The corresponding ratio estimator is

\[
\hat{r}_0(s,a)=\exp(\ell_0(s,a)).
\]

This estimator is deterministic conditional on \((s,a)\), so

\[
\operatorname{Var}[\hat{r}_0(s,a)\mid s,a] = 0.
\]

In general, however,

\[
\hat{r}_0(s,a) \neq r_{\mathrm{act}}(s,a),
\]

unless the fiber distribution collapses to \(u=0\) or the ratio is locally flat in the fiber direction.

If \(\ell(s,a,u)=\log r_{\mathrm{dum}}(s,a,u)\) admits a local expansion

\[
\ell(s,a,u)
=
\ell_0 + g^\top u + \frac{1}{2}u^\top H u + O(\|u\|^3),
\]

and \(U \sim q_{\theta_{\text{old}}}(\cdot\mid a,s)\) has mean \(\mu\) and covariance \(\Sigma\), then

\[
r_{\mathrm{act}}(s,a)-\hat{r}_0(s,a)
=
O(\|g\|\,\|\mu\| + \operatorname{tr}(H\Sigma)).
\]

So `GenPOU0Clip` is a biased but zero-conditional-variance approximation.

## 6. KL Decomposition

The dummy KL splits cleanly into action and fiber terms:

\[
D_{\mathrm{KL}}^{\mathrm{dum}}
=
\mathbb{E}_{(a,u)\sim \bar{\pi}_{\theta_{\text{old}}}}
\left[
\log \frac{\bar{\pi}_{\theta_{\text{old}}}(a,u\mid s)}{\bar{\pi}_\theta(a,u\mid s)}
\right]
=
D_{\mathrm{KL}}^{\mathrm{act}}
+
\mathbb{E}_{a\sim \pi_{\theta_{\text{old}}}^g}
D_{\mathrm{KL}}\!\left(q_{\theta_{\text{old}}}(\cdot\mid a,s)\,\|\,q_\theta(\cdot\mid a,s)\right).
\]

The second term reflects internal rearrangement inside the doubled dummy representation. This is why `section-KL` and proposal-based pushforward KL are better aligned with the actual proximal change in the executed action measure.

## 7. Implemented Trust Region

Both new algorithms use log-domain clipping:

\[
\ell_{\mathrm{clip}}=\operatorname{clamp}(\ell, -\delta, \delta),
\qquad
r_{\mathrm{clip}}=\exp(\ell_{\mathrm{clip}}).
\]

The PPO-style surrogate then becomes

\[
\min\left(r \hat{A}, r_{\mathrm{clip}} \hat{A}\right),
\qquad r=\exp(\ell).
\]

This keeps the trust region symmetric in log-likelihood space.

## 8. Algorithm Mapping in This Repository

- `GenPO`:
  uses dummy-action ratios from stored rollout samples.
- `GenPOU0Clip`:
  replaces dummy-action ratios with the diagonal section estimator \(\ell_0\), and uses `section_kl = -\ell_0` for adaptive learning-rate control.
- `GenPOPFClip`:
  estimates the pushforward log ratio with antithetic fiber proposals:

\[
\hat{\ell}_{\mathrm{pf}}
=
\log \frac{1}{2M} \sum_{m,\pm}
\exp\left(
\frac{1}{2}\|z_{\text{old}}^{(m,\pm)}\|^2
- \frac{1}{2}\|z_\theta^{(m,\pm)}\|^2
\right).
\]

This is a proposal-based approximation and is not claimed to be an exact unbiased estimator of the pushforward ratio.
