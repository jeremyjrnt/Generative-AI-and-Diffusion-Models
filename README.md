# Generative AI and Diffusion Models - Course Projects

This repository contains four projects implementing foundational concepts and advanced techniques in diffusion models and score-based generative modeling. Each project builds upon theoretical foundations from seminal papers in the field, progressing from simple 1D/2D examples to practical image generation applications.

---

## Project 1: Score Matching and Langevin Dynamics

### Objectives
The first project introduces **score-based generative modeling** by implementing score matching and Langevin dynamics on a 2D Gaussian Mixture Model (GMM). The primary goals are to:
- Compute the analytical score function (gradient of log-density) for a GMM
- Train a neural network to approximate the score function
- Generate samples using Langevin Dynamics
- Implement Annealed Langevin Dynamics for improved sampling
- Extend these concepts to image generation on the MNIST dataset

### Methods and Implementation

#### Part 1: Analytical Score Computation
I implemented closed-form computations for the density and score of a 2D GMM:

**Data Distribution:**
- GMM with two components: μ₁ = [1, 1], μ₂ = [-1, -1]
- Standard deviations: σ₁ = σ₂ = √0.1
- Mixing probability: p = 0.15 (15% from first Gaussian, 85% from second)

**Score Function Formula:**
```
∇ₓ log p(x) = Σₖ wₖ(x) · ∇ₓ log N(x|μₖ, σₖ²I)
```
where wₖ(x) are the posterior responsibilities (normalized weights) of each component.

The score vectors point toward high-density regions, visualized using heatmaps for density and quiver plots for the score field.

#### Part 2: Neural Score Network Training
I trained a simple Multi-Layer Perceptron (MLP) to approximate the score function:

**Network Architecture:**
- Input: 2D points (x, y)
- Hidden layers: [128, 128] with ReLU activations
- Output: 2D score vector

**Training Configuration:**
- Training samples: Generated from the GMM distribution
- Loss function: Mean Squared Error (MSE) between predicted and true scores
- Optimizer: Adam
- Learning rate: Configured based on the training procedure

**Key Insight:** The trained model struggles in low-density regions due to training data imbalance—most samples come from the dominant mode (85%), leading to underfitting in underrepresented areas.

#### Part 3: Sampling with Langevin Dynamics
I implemented **Langevin Dynamics** to generate samples using the trained score model:

**Algorithm:**
```
x_{t+1} = x_t + ε·∇ₓ log p(x_t) + √(2ε)·z_t
```
where z_t ~ N(0, I)

**Hyperparameters:**
- Step size (ε): 0.01
- Number of steps: 10,000
- Initial distribution: Random Gaussian noise

#### Part 4: Annealed Langevin Dynamics
To address the limitation of Langevin dynamics in low-density regions, I implemented **Annealed Langevin Dynamics** which uses multiple noise scales:

**Procedure:**
1. Define a sequence of decreasing noise levels: σ₁ > σ₂ > ... > σ_L
2. For each noise level, perturb the data distribution and run Langevin dynamics
3. Gradually anneal from high to low noise, allowing exploration of the full space

This multi-scale approach enables sampling from all modes of the distribution, not just the dominant one.

#### Part 5: Image Generation on MNIST
I extended the score matching approach to generate MNIST handwritten digits:

**Network Architecture:**
- Input: 28×28 flattened images + noise level embedding
- Hidden layers with appropriate capacity for image modeling
- Output: 784-dimensional score vector

**Training:**
- Loss function: Denoising Score Matching with annealed noise schedule
- Learning rate: 1e-3
- Data: MNIST handwritten digits dataset

**Sampling:**
- Method: Annealed Langevin Dynamics
- Generates recognizable handwritten digits from random noise

### Results and Insights
- **Analytical vs. Learned Scores:** The neural network approximates the true score well in high-density regions but fails in sparse areas
- **Sampling Quality:** Standard Langevin dynamics may miss minor modes; Annealed Langevin dynamics successfully samples from all modes
- **MNIST Generation:** Successfully generated recognizable digit images, demonstrating the scalability of score-based methods to high-dimensional data

### Theory
This project is based on **score matching** and **Langevin dynamics** from:
- **Song, Y., & Ermon, S. (2019).** "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS 2019.*
- The score function ∇ₓ log p(x) characterizes the data distribution without requiring normalization constants
- Langevin dynamics provides a MCMC sampling procedure that follows the score field
- Annealed Langevin dynamics addresses the challenge of mixing in multi-modal distributions

---

## Project 2: Denoising Diffusion Probabilistic Models (DDPM)

### Objectives
This project implements the core **DDPM** framework on a 1D Gaussian Mixture Model to understand:
- The forward diffusion process (adding noise gradually)
- The mathematical formulation of the noising schedule
- How the distribution evolves through diffusion timesteps
- Visualization of the diffusion trajectory

### Methods and Implementation

#### Forward Diffusion Process
I implemented the DDPM forward process that progressively adds Gaussian noise to data:

**Mathematical Formulation:**
```
q(x_t | x_{t-1}) = N(x_t; √α·x_{t-1}, β)
q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)I)
```

where:
- α = 1 - β (noise retention coefficient)
- ᾱ_t = ∏ᵢ₌₁ᵗ αᵢ (cumulative product)

**Hyperparameters:**
- Total timesteps (T): 250
- Noise level (β): 0.02 (constant schedule)
- α = 1 - β = 0.98
- ᾱ_t = αᵗ (exponential decay)

**Initial Data Distribution:**
- 1D Gaussian Mixture: N(-4, 1) and N(4, 1) with equal mixing weights
- This bimodal distribution gradually transforms into a standard Gaussian

#### Implementation Details
The `do_diffusion()` function implements equation (2) from the DDPM paper:

```python
def do_diffusion(data, steps=TIME_STEPS, beta=BETA):
    distributions, samples = [None], [data]
    xt = data
    for t in range(1, steps + 1):
        mean = sqrt_alpha * xt
        std = sqrt_beta
        dist = torch.distributions.Normal(mean, std)
        xt = dist.rsample()  # Reparameterization trick
        distributions.append(dist)
        samples.append(xt)
    return distributions, samples
```

### Results and Insights

**Visualization:**
- **Histogram at t=0:** Clear bimodal distribution with peaks at ±4
- **Diffusion Trajectory:** Plotted the evolution of sample distributions across all 250 timesteps
- **Final Distribution (t=250):** Converges to approximately N(0, I)

**Key Observations:**
1. The two modes gradually merge and spread out
2. By t ≈ 100-150, the distribution becomes unimodal
3. At t = 250, the distribution is nearly indistinguishable from Gaussian noise
4. The process is deterministic given the noise schedule

### Theory
This project implements concepts from:
- **Ho, J., Jain, A., & Abbeel, P. (2020).** "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.*

**Key Theoretical Concepts:**
- **Forward Process:** A fixed Markov chain that gradually adds noise according to a variance schedule β₁, ..., β_T
- **Reparameterization:** x_t can be sampled directly from x_0 using the closed form with ᾱ_t
- **Variance Schedule:** Controls the rate of noise addition; linear schedules are common
- **Reverse Process:** Not implemented in this project but prepares understanding for denoising

---

## Project 3: Score-Based Generative Modeling via Stochastic Differential Equations

### Objectives
This project extends discrete diffusion models to continuous time using **Stochastic Differential Equations (SDEs)**. Goals include:
- Understanding the SDE formulation of diffusion processes
- Training a score network for continuous-time modeling
- Implementing forward and reverse SDE/ODE solvers
- Comparing SDE vs. ODE sampling trajectories

### Methods and Implementation

#### SDE Formulation
I implemented the **Variance Preserving (VP) SDE**, which is the continuous-time limit of DDPM:

**Forward SDE:**
```
dx_t = -½β(t)x_t dt + √β(t) dw_t
```

**Reverse SDE:**
```
dx_t = [-½β(t)x_t - β(t)∇ₓ log p_t(x_t)] dt + √β(t) dw̄_t
```

**ODE Formulation (Probability Flow ODE):**
```
dx_t = [-½β(t)x_t - ½β(t)∇ₓ log p_t(x_t)] dt
```

**Hyperparameters:**
- β₀ = 0.1 (initial diffusion coefficient)
- β₁ = 20 (final diffusion coefficient)
- β(t) = β₀ + (β₁ - β₀)t (linear schedule)
- Time horizon: t ∈ [0, 1]

#### Data Distribution
- **Initial Distribution:** Mixture of two Gaussians
  - Mode 1: μ = -3, σ = 0.5
  - Mode 2: μ = 3, σ = 0.5
  - Equal mixing probability (50/50)

#### Score Network Training
**Architecture:**
```python
class ScoreNetwork(nn.Module):
    - Input: (x, t) where x ∈ ℝ and t ∈ [0,1]
    - Layer 1: Linear(2, 64) + SiLU
    - Layer 2: Linear(64, 64) + SiLU
    - Output: Linear(64, 1)
```

**Training Configuration:**
- Training steps: 3000
- Batch size: 2048
- Optimizer: Adam
- Learning rate: 1e-3
- Loss: MSE between predicted and true scores (computed analytically)

#### Simulation Procedures

**Forward SDE Simulation:**
- Discretization: Euler-Maruyama method
- Number of steps: 1000
- Time step: Δt = 1e-3
- Samples: 100,000 particles
- Visualizes how the bimodal distribution diffuses into Gaussian noise

**Forward ODE Simulation:**
- Same discretization scheme
- Samples: 10,000 particles
- Deterministic trajectories (no Brownian motion term)

**Reverse SDE Sampling:**
- Starts from N(0, I) at t=1
- Uses learned score function ∇ₓ log p_t(x_t)
- Samples: 10,000 particles
- Stochastic trajectories back to data distribution

**Reverse ODE Sampling (Bonus):**
- Deterministic denoising using probability flow ODE
- Samples: 10,000 particles
- Smoother trajectories compared to SDE

### Results and Insights

**Visualization:**
I created a 2×2 grid of heatmap plots showing density evolution:
1. **Forward SDE:** Bimodal → Gaussian diffusion with stochastic paths
2. **Forward ODE:** Smooth deterministic evolution
3. **Reverse SDE:** Noise → bimodal distribution with high-variance trajectories
4. **Reverse ODE:** Clean deterministic generation

**Key Findings:**
- **SDE vs. ODE:** SDEs produce more diverse samples due to stochasticity; ODEs give deterministic mappings
- **Score Function Quality:** The learned score network successfully captures both modes despite their separation
- **Sampling Speed:** ODE sampling can use larger step sizes (faster) while maintaining quality
- **Trajectory Smoothness:** ODE paths are continuous and interpretable; SDE paths show random fluctuations

### Theory
This project implements concepts from:
- **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).** "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021.*

**Theoretical Foundations:**
- **SDE Framework:** Diffusion processes are SDEs; the reverse process is also an SDE that depends on the score function
- **Probability Flow ODE:** There exists a deterministic ODE that generates the same marginal distributions as the SDE
- **Score Matching:** The score function ∇ₓ log p_t(x_t) is learned via denoising score matching
- **Variance Preserving:** This SDE formulation maintains constant variance, corresponding to DDPM in continuous time
- **Numerical Integration:** Euler-Maruyama for SDEs, Euler method for ODEs

---

## Project 4: Practical Diffusion Applications

Project 4 consists of two parts: 2D plot diffusion and image diffusion with advanced techniques.

### Part 1: 2D Plot Diffusion

#### Objectives
- Implement complete DDPM pipeline for 2D toy datasets
- Train a noise prediction network ε_θ(x_t, t)
- Implement both DDPM and DDIM sampling
- Understand the trade-off between sampling quality and speed

#### Methods and Implementation

**Network Architecture (SimpleNet):**
- Uses `TimeLinear` layers that modulate features based on timestep
- Time embedding: Sinusoidal embeddings (similar to Transformer positional encoding)
- Hidden dimensions: [256, 256, 256]
- Activation: SiLU (Sigmoid Linear Unit) for smooth gradients
- Input: 2D points + time embedding
- Output: 2D noise prediction

**DDPM Components:**

1. **Variance Scheduler:**
   - Linear schedule: β linearly increases from β₁=1e-4 to β_T=0.02
   - Quadratic schedule option: β = (√β₁ to √β_T)²
   - Precomputes α, ᾱ for efficient sampling

2. **Forward Process (q_sample):**
   ```python
   x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
   ```
   - Direct sampling from q(x_t|x_0) without iterating through all steps

3. **Reverse Process (p_sample):**
   ```python
   x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t,t)) + σ_t·z
   ```
   - Denoises one step at a time
   - σ_t: posterior variance (small or large parameterization)
   - No noise added when t=0

4. **Training Loss:**
   ```python
   L = ||ε - ε_θ(x_t, t)||²
   ```
   - Simplified noise matching objective from the DDPM paper

**DDIM Sampling:**
- **Non-Markovian** deterministic sampling process
- Allows variable number of inference steps (e.g., 50 instead of 1000)
- Controlled stochasticity via η parameter:
  - η = 0: fully deterministic (ODE)
  - η = 1: equivalent to DDPM
- **Acceleration:** Can generate samples 20× faster with minimal quality loss

**Implementation Details:**
```python
def ddim_p_sample(xt, t, t_prev, eta=0.0):
    # Predict x_0 from x_t
    x0_pred = (xt - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
    
    # Direction pointing to x_t
    dir_xt = √(1-ᾱ_{t-1}-σ²)·ε_θ
    
    # DDIM update
    x_{t-1} = √ᾱ_{t-1}·x0_pred + dir_xt + σ·z
```

#### Results and Insights
- **Chamfer Distance:** Used to measure quality of generated 2D point clouds against ground truth
- **DDPM vs. DDIM:** DDIM achieves comparable quality with 20-50 steps vs. DDPM's 1000 steps
- **Deterministic Sampling:** η=0 enables interpolation in latent space (useful for manipulation)

---

### Part 2: Image Diffusion on AFHQ Dataset

#### Objectives
- Scale DDPM to realistic image generation (32×32 resolution)
- Implement U-Net architecture for image denoising
- Add classifier-free guidance for conditional generation
- Train on Animal Faces-HQ (AFHQ) dataset with three classes: cat, dog, wild

#### Methods and Implementation

**U-Net Architecture:**
- **Backbone:** Encoder-decoder structure with skip connections
- **Time Embedding:** Sinusoidal embeddings → MLP(4×channels)
- **Channel Configuration:** base_ch=128, multipliers=[1, 2, 2, 2]
- **Attention Mechanisms:** Multi-head self-attention at resolution 1 (lowest spatial resolution)
- **Residual Blocks:** 4 ResBlocks per resolution level
- **Normalization:** GroupNorm (32 groups)
- **Activation:** Swish (SiLU)
- **Dropout:** 0.1 for regularization

**Architecture Flow:**
1. **Encoder (Downsampling):**
   - Input: 3×32×32 RGB image
   - Conv layer → [128, 256, 256, 256] channels
   - Downsampling between levels
   - Skip connections stored for decoder

2. **Middle Block:**
   - 2 ResBlocks with self-attention
   - Bottleneck: 256 channels

3. **Decoder (Upsampling):**
   - Concatenate skip connections
   - [256, 256, 256, 128] channels
   - Upsampling between levels
   - Output: 3×32×32 noise prediction

**Classifier-Free Guidance (CFG):**

CFG enables conditional generation without a separate classifier:

**Training Procedure:**
```python
# Randomly drop class labels with probability cfg_dropout
if random() < 0.1:
    class_label = 0  # Null class
    
# Combine time and class embeddings
temb = time_embedding(t) + class_embedding(class_label)
```

**Sampling with Guidance:**
```python
# Predict noise both conditionally and unconditionally
ε_cond = network(xt, t, class_label)
ε_uncond = network(xt, t, null_class)

# Apply guidance
ε = ε_uncond + w·(ε_cond - ε_uncond)
```

**Hyperparameters:**
- Guidance scale (w): User-defined (typically 1.0-7.5)
- Higher w → stronger conditioning, less diversity
- cfg_dropout: 0.1 (10% unconditional training)

**Training Configuration:**
- **Batch size:** 8 (adjustable to 10-16 with more GPU memory)
- **Training steps:** 50,000
- **Learning rate:** 2e-4 with warmup
- **Warmup steps:** 200
- **Optimizer:** Adam
- **LR Schedule:** Linear warmup then constant
- **Logging interval:** Every 2000 steps
- **Dataset:** AFHQ (max 3000 images per category)

**DDPMScheduler Parameters:**
- num_train_timesteps: 1000
- β₁: 1e-4
- β_T: 0.02
- Schedule: Linear
- Sigma type: "small" (σ² = β̃_t, posterior variance)

**Variance Formulation:**
```python
σ_t² = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t  # small variance
```

**Forward Diffusion (add_noise):**
```python
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
```

**Reverse Step:**
```python
μ_t = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ)
x_{t-1} = μ_t + σ_t·z  (z=0 if t=0)
```

#### Results and Insights

**Generation Quality:**
- Successfully generates 32×32 animal face images
- Recognizable features (eyes, fur patterns, ears)
- Class conditioning works effectively with CFG

**Classifier-Free Guidance Benefits:**
- **No separate classifier needed:** Simplifies training pipeline
- **Controllable trade-off:** Balance between fidelity and diversity via guidance scale w
- **Better sample quality:** Typically outperforms classifier guidance

**Training Observations:**
- Loss steadily decreases over 50k steps
- Sample quality improves significantly after ~10k steps
- Checkpoint saving enables resuming and evaluation

**FID Evaluation:**
- Fréchet Inception Distance measures generation quality
- Lower FID = better quality
- Inception network pretrained on AFHQ for feature extraction

### Theory
Part 1 and 2 build upon:
- **Ho, J., Jain, A., & Abbeel, P. (2020).** "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.*
  - DDPM formulation, variance schedules, simplified training objective
  
- **Song, J., Meng, C., & Ermon, S. (2021).** "Denoising Diffusion Implicit Models." *ICLR 2021.*
  - DDIM non-Markovian sampling, acceleration techniques
  
- **Ho, J., & Salimans, T. (2022).** "Classifier-Free Diffusion Guidance." *NeurIPS 2022 Workshop.*
  - CFG training and sampling, conditional generation without classifiers

**Key Theoretical Concepts:**

1. **Noise Prediction Parameterization:**
   - Instead of predicting x₀ or score, predict noise ε
   - Equivalent formulations but better empirical performance

2. **U-Net for Images:**
   - Skip connections preserve spatial information
   - Multi-scale processing captures both local and global features
   - Attention at coarse resolution for global coherence

3. **Variance Schedule:**
   - Linear schedule works well for images
   - Cosine schedule (not used here) can improve quality

4. **Classifier-Free Guidance:**
   - Implicit classifier: p(c|x) ∝ p(x|c)/p(x)
   - Score modification: ∇log p(x|c) + w·(∇log p(x|c) - ∇log p(x))
   - Trained jointly in single model using null conditioning

---

## Summary

These four projects provide a comprehensive journey through diffusion models:

1. **Project 1** establishes the foundations of score-based modeling with analytical solutions
2. **Project 2** introduces the discrete DDPM framework and forward diffusion
3. **Project 3** extends to continuous-time SDEs and explores SDE/ODE sampling
4. **Project 4** applies these concepts to practical 2D and image generation tasks with state-of-the-art techniques

### Key Papers Referenced

1. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS 2019.*
2. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.*
3. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021.*
4. Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models." *ICLR 2021.*
5. Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." *NeurIPS 2022 Workshop.*

### Technical Skills Developed

- Implementing score matching and Langevin dynamics
- Forward and reverse diffusion processes
- SDE/ODE numerical integration
- Neural network architectures for score/noise prediction
- U-Net with attention mechanisms
- Conditional generation with classifier-free guidance
- Efficient sampling with DDIM
- PyTorch implementations of diffusion models
- Evaluation metrics (Chamfer distance, FID)
