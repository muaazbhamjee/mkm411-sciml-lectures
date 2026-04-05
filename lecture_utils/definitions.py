"""
lecture_utils/definitions.py
=============================
Concept definition callout boxes for lecture notebooks.

Each definition is formatted as a styled markdown string that renders as
a prominent callout. Definitions are cited to primary literature.

Usage
-----
From a markdown cell in the notebook, import the string directly:

    from lecture_utils.definitions import DEF_ACTIVATION, REFERENCES_L1

Or use the helper:

    from lecture_utils.definitions import define
    define('activation_function')   # prints the markdown string

References
----------
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
    https://www.deeplearningbook.org

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training
    deep feedforward neural networks. AISTATS 2010, 249-256.
    http://proceedings.mlr.press/v9/glorot10a

Kingma, D.P., & Ba, J. (2015). Adam: A method for stochastic optimization.
    ICLR 2015. https://arxiv.org/abs/1412.6980

Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed
    neural networks. Journal of Computational Physics, 378, 686-707.
    https://doi.org/10.1016/j.jcp.2018.10.045

Raissi, M., Yazdani, A., & Karniadakis, G.E. (2020). Hidden fluid mechanics.
    Science, 367(6481), 1026-1030.
    https://doi.org/10.1126/science.aaw4741
"""

# ── Lecture 1 definitions ─────────────────────────────────────────────────────

DEF_NEURON = """\
> **Definition — Neuron**  
> The fundamental computational unit of a neural network. A single neuron
> computes a weighted sum of its inputs, adds a bias, and passes the result
> through an activation function:
> $a = \\sigma\\!\\left(\\sum_j w_j x_j + b\\right)$  
> Biologically inspired but mathematically a simple non-linear function.  
> *(Goodfellow et al., 2016, §6.1)*
"""

DEF_LAYER = """\
> **Definition — Layer**  
> A collection of neurons that operate in parallel on the same input.
> Each layer transforms its input vector $\\mathbf{a}^{(l-1)}$ into an output
> vector $\\mathbf{a}^{(l)}$ via $\\mathbf{a}^{(l)} = \\sigma(W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)})$.
> Layers are stacked sequentially to form the network.  
> *(Goodfellow et al., 2016, §6.1)*
"""

DEF_WEIGHTS_BIAS = """\
> **Definition — Weights and Bias**  
> **Weights** $W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}$ control the strength of connections
> between layers. **Bias** $\\mathbf{b}^{(l)} \\in \\mathbb{R}^{n_l}$ shifts the activation
> threshold independently of the input. Together they are the **learnable parameters**
> $\\theta$ of the network — adjusted during training to minimise the loss.  
> *(Goodfellow et al., 2016, §6.1)*
"""

DEF_DEPTH_WIDTH = """\
> **Definition — Depth and Width**  
> **Depth** is the number of hidden layers. **Width** is the number of neurons
> per hidden layer. Depth allows the network to learn hierarchical representations
> — early layers capture simple features, deeper layers combine them into complex
> patterns. The Universal Approximation Theorem guarantees that sufficient width
> in a single hidden layer can approximate any continuous function, but depth is
> far more parameter-efficient in practice.  
> *(Goodfellow et al., 2016, §6.4)*
"""

DEF_ACTIVATION = """\
> **Definition — Activation Function**  
> A non-linear function $\\sigma(\\cdot)$ applied element-wise to the pre-activation
> $\\mathbf{z}^{(l)} = W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}$.
> Without it, any stack of linear layers collapses to a single linear transformation
> regardless of depth — the network could only represent linear functions.
> Common choices: `tanh`, `ReLU`, `sigmoid`, `ELU`.
> **For PINNs:** must be at least twice differentiable — `tanh` is the standard choice.  
> *(Goodfellow et al., 2016, §6.3)*
"""

DEF_FORWARD_PASS = """\
> **Definition — Forward Pass**  
> The sequential computation from input to output: each layer receives the
> activations of the previous layer, applies its weights, bias, and activation
> function, and passes the result forward. For a network with $L$ layers:
> $f_\\theta(\\mathbf{x}) = \\mathbf{a}^{(L)} = \\sigma^{(L)}(W^{(L)} \\cdots \\sigma^{(1)}(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}) \\cdots + \\mathbf{b}^{(L)})$  
> *(Goodfellow et al., 2016, §6.1)*
"""

DEF_XAVIER = """\
> **Definition — Xavier / Glorot Initialisation**  
> A weight initialisation strategy that scales random weights to preserve
> signal variance across layers: $\\mathrm{Var}(w) = 2/(n_{\\mathrm{in}} + n_{\\mathrm{out}})$.
> Prevents signals from vanishing (too small) or exploding (too large) as they
> propagate through deep networks. Essential for stable training with `tanh`.  
> *(Glorot & Bengio, 2010)*
"""

DEF_LOSS = """\
> **Definition — Loss Function**  
> A scalar measure $\\mathcal{L}(\\theta)$ of how poorly the network predictions
> match the targets (or, for PINNs, how poorly the PDE is satisfied).
> Training minimises $\\mathcal{L}$ by adjusting $\\theta$.
> Common choice for regression: **Mean Squared Error (MSE)**
> $\\mathcal{L} = \\frac{1}{N}\\sum_{i=1}^N (f_\\theta(\\mathbf{x}_i) - y_i)^2$.  
> *(Goodfellow et al., 2016, §6.2)*
"""

DEF_GRADIENT_DESCENT = """\
> **Definition — Gradient Descent**  
> An iterative optimisation algorithm that updates parameters in the direction
> of steepest descent of the loss:
> $\\theta \\leftarrow \\theta - \\eta \\nabla_\\theta \\mathcal{L}$
> where $\\eta > 0$ is the **learning rate**. Requires differentiability of
> $\\mathcal{L}$ with respect to $\\theta$ — guaranteed by autograd.  
> *(Goodfellow et al., 2016, §4.3)*
"""

DEF_LEARNING_RATE = """\
> **Definition — Learning Rate** $\\eta$  
> Controls the step size in gradient descent. Too large: overshoots minima,
> training diverges. Too small: converges very slowly, may get stuck.
> Adaptive methods (Adam) adjust $\\eta$ per parameter automatically.  
> *(Goodfellow et al., 2016, §8.3)*
"""

DEF_BACKPROP = """\
> **Definition — Backpropagation**  
> An efficient algorithm for computing $\\nabla_\\theta \\mathcal{L}$ by applying
> the chain rule backwards through the network. Defines error signals
> $\\delta^{(l)} = \\partial \\mathcal{L} / \\partial \\mathbf{z}^{(l)}$ propagated
> from output to input, then uses them to compute weight gradients
> $\\nabla_{W^{(l)}} \\mathcal{L} = \\delta^{(l)}(\\mathbf{a}^{(l-1)})^T$.
> In PyTorch, `.backward()` executes this automatically via **autograd**.  
> *(Goodfellow et al., 2016, §6.5)*
"""

DEF_OPTIMISER = """\
> **Definition — Optimiser**  
> The algorithm that uses gradients to update network parameters.
> **SGD**: fixed learning rate, noisy updates.
> **Adam** *(Kingma & Ba, 2015)*: maintains per-parameter adaptive learning rates
> using first moment $m_t$ (mean) and second moment $v_t$ (variance) of gradients:
> $\\theta_t = \\theta_{t-1} - \\eta\\,\\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon)$.
> Adam is the standard choice for PINNs.  
> *(Kingma & Ba, 2015)*
"""

DEF_MINIBATCH = """\
> **Definition — Mini-batch and Epoch**  
> A **mini-batch** is a random subset of $B$ training samples used to compute
> one gradient update — a compromise between noisy single-sample (SGD) and
> expensive full-dataset updates. An **epoch** is one complete pass through the
> training dataset (i.e. $N/B$ mini-batch updates). Mini-batch training is
> standard in deep learning — it is memory-efficient, GPU-parallelisable,
> and the gradient noise helps escape local minima.  
> *(Goodfellow et al., 2016, §8.1)*
"""

DEF_OVERFITTING = """\
> **Definition — Overfitting and Underfitting**  
> **Overfitting**: the model memorises training data, including its noise,
> and fails to generalise — high training accuracy, poor test accuracy.
> **Underfitting**: the model is too simple to capture the underlying pattern
> — poor performance on both training and test data.
> Detected by monitoring a **validation set** — data held out from training
> and used only for evaluation. A growing train/val loss gap signals overfitting.  
> *(Goodfellow et al., 2016, §5.2)*
"""

DEF_REGULARISATION = """\
> **Definition — Regularisation**  
> Techniques that reduce overfitting by constraining the model:
> - **L2 (weight decay)**: adds $\\lambda\\|\\theta\\|^2$ to the loss, penalising large weights
> - **Dropout**: randomly zeros neurons during training, preventing co-adaptation
> - **Early stopping**: halts training when validation loss begins to increase
> All three are ways of reducing effective model complexity without reducing depth or width.  

> **Co-adaptation** occurs when neurons learn to rely on the presence of specific
> other neurons to correct their errors, rather than learning independently useful
> features. The network effectively memorises a set of neuron partnerships that work
> well on training data but fail to generalise. Dropout breaks co-adaptation by
> randomly disabling neurons during each training step — no neuron can depend on
> any other being present, so each is forced to learn robust features independently.

> *(Goodfellow et al., 2016, §7)*
"""

# ── Lecture 2 definitions ─────────────────────────────────────────────────────

DEF_COLLOCATION = """\
> **Definition — Collocation Points**  
> Randomly sampled points in the space-time domain at which the PDE residual
> is evaluated during PINN training. No solution data is needed at these points
> — only the governing equation must be satisfied. The number and distribution
> of collocation points affects training stability: too few leads to underdetermined
> physics; fixed points can cause overfitting to specific locations (mitigated by
> periodic resampling).  
> *(Raissi et al., 2019, §2)*
"""

DEF_PDE_RESIDUAL = """\
> **Definition — PDE Residual**  
> For a PDE of the form $\\mathcal{N}[u] = 0$, the residual at a point $\\mathbf{x}$
> is $r(\\mathbf{x}) = \\mathcal{N}[f_\\theta](\\mathbf{x})$ — how far the network
> prediction $f_\\theta$ is from satisfying the equation at that point.
> The PINN loss minimises the mean squared residual over all collocation points:
> $\\mathcal{L}_{\\mathrm{pde}} = \\frac{1}{N_c}\\sum_{i=1}^{N_c} |r(\\mathbf{x}_i)|^2$.  
> *(Raissi et al., 2019, §2)*
"""

DEF_AUTOGRAD = """\
> **Definition — Automatic Differentiation (Autograd)**  
> A computational technique for evaluating exact derivatives of functions
> defined by computer programs. PyTorch's autograd builds a computational
> graph during the forward pass and traverses it in reverse to compute gradients
> via the chain rule. In PINNs, autograd computes spatial and temporal derivatives
> of the network output with respect to its inputs — enabling the PDE residual
> to be evaluated without discretisation or finite differences.  
> *(Goodfellow et al., 2016, §6.5.1)*
"""

DEF_LOSS_WEIGHTS = """\
> **Definition — Loss Weights** $\\lambda_{\\mathrm{pde}},\\, \\lambda_{\\mathrm{bc}},\\, \\lambda_{\\mathrm{ic}}$  
> Scalar coefficients that balance competing terms in the PINN composite loss.
> Higher $\\lambda_{\\mathrm{bc}}$ and $\\lambda_{\\mathrm{ic}}$ force the network to
> satisfy boundary and initial conditions before minimising the PDE residual —
> preventing the trivial solution (constant output, zero PDE residual) from
> dominating early training. Choosing appropriate weights is an active research
> area; adaptive weighting schemes exist but manual tuning remains common.  
> *(Raissi et al., 2019, §2)*
"""

DEF_INVERSE_PROBLEM = """\
> **Definition — Inverse Problem**  
> A problem in which unknown quantities (fields, parameters, boundary conditions)
> are inferred from observable data, reversing the usual forward problem direction.
> Classical CFD solves the **forward problem**: given governing equations and BCs,
> find the flow field. PINNs solve the **inverse problem**: given sparse observations
> of one field (e.g. concentration $c$), infer all other fields ($u$, $v$, $p$)
> by simultaneously satisfying multiple coupled PDEs.
> This is one of the most powerful and distinctive capabilities of PINNs.  
> *(Raissi et al., 2020, §2)*
"""

# ── Reference sections ────────────────────────────────────────────────────────

REFERENCES_L1 = """\
---
## References

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.  
https://www.deeplearningbook.org  
*(Free online — the standard reference for neural network fundamentals)*

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep
feedforward neural networks. *Proceedings of AISTATS 2010*, 249–256.  
http://proceedings.mlr.press/v9/glorot10a

Kingma, D.P., & Ba, J. (2015). Adam: A method for stochastic optimization.
*ICLR 2015*. https://arxiv.org/abs/1412.6980

Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural
networks: A deep learning framework for solving forward and inverse problems
involving nonlinear PDEs. *Journal of Computational Physics*, 378, 686–707.  
https://doi.org/10.1016/j.jcp.2018.10.045
"""

REFERENCES_L2 = """\
---
## References

Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural
networks: A deep learning framework for solving forward and inverse problems
involving nonlinear PDEs. *Journal of Computational Physics*, 378, 686–707.  
https://doi.org/10.1016/j.jcp.2018.10.045

Raissi, M., Yazdani, A., & Karniadakis, G.E. (2020). Hidden fluid mechanics:
Learning velocity and pressure fields from flow visualizations. *Science*,
367(6481), 1026–1030. https://doi.org/10.1126/science.aaw4741

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.  
https://www.deeplearningbook.org

Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G.E. (2021). Learning
nonlinear operators via DeepONet. *Nature Machine Intelligence*, 3, 218–229.  
https://doi.org/10.1038/s42256-021-00302-5

Li, Z. et al. (2021). Fourier neural operator for parametric partial differential
equations. *ICLR 2021*. https://arxiv.org/abs/2010.08895

Jaeger, H. (2001). The echo state approach to analysing and training recurrent
neural networks. *GMD Report 148*. German National Research Center for Information
Technology.

Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N.,
Singhal, U., Ramamoorthi, R., Barron, J.T., & Ng, R. (2020). Fourier features let
networks learn high frequency functions in low dimensional domains.
*Advances in Neural Information Processing Systems*, 33, 7537–7547.
https://arxiv.org/abs/2006.10739
"""

# ── Helper ────────────────────────────────────────────────────────────────────

_ALL_DEFS = {
    'neuron':           DEF_NEURON,
    'layer':            DEF_LAYER,
    'weights_bias':     DEF_WEIGHTS_BIAS,
    'depth_width':      DEF_DEPTH_WIDTH,
    'activation':       DEF_ACTIVATION,
    'forward_pass':     DEF_FORWARD_PASS,
    'xavier':           DEF_XAVIER,
    'loss':             DEF_LOSS,
    'gradient_descent': DEF_GRADIENT_DESCENT,
    'learning_rate':    DEF_LEARNING_RATE,
    'backprop':         DEF_BACKPROP,
    'optimiser':        DEF_OPTIMISER,
    'minibatch':        DEF_MINIBATCH,
    'overfitting':      DEF_OVERFITTING,
    'regularisation':   DEF_REGULARISATION,
    'collocation':      DEF_COLLOCATION,
    'pde_residual':     DEF_PDE_RESIDUAL,
    'autograd':         DEF_AUTOGRAD,
    'loss_weights':     DEF_LOSS_WEIGHTS,
    'inverse_problem':  DEF_INVERSE_PROBLEM,
}


def define(concept):
    """Print the markdown definition for a concept. For use in notebooks."""
    from IPython.display import display, Markdown
    if concept not in _ALL_DEFS:
        raise KeyError(f"Unknown concept '{concept}'. "
                       f"Available: {sorted(_ALL_DEFS.keys())}")
    display(Markdown(_ALL_DEFS[concept]))
