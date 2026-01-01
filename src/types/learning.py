# Copyright (C) 2025 Gil Benezer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Learning and Data-Driven Control Types

Result types for machine learning and data-driven control methods:
- Neural network training
- Reinforcement learning (RL)
- Imitation learning
- Online adaptation

These methods learn control policies and dynamics models from data.

Mathematical Background
----------------------
Supervised Learning (Dynamics Models):
    Given data: {(x[k], u[k], x[k+1])}
    Learn: f̂ such that x[k+1] ≈ f̂(x[k], u[k])

    Loss: L = (1/N) Σ ||x[k+1] - f̂(x[k], u[k])||²

    Neural ODE: dx/dt = f_θ(x, u)
    Optimize θ via backpropagation through ODE solver

Reinforcement Learning:
    MDP: (S, A, P, R, γ)
    Policy: π(a|s) or π_θ(s) → a
    Value: V^π(s) = E[Σ γ^t r_t | s_0=s, π]
    Q-function: Q^π(s,a) = E[Σ γ^t r_t | s_0=s, a_0=a, π]

    Optimal policy: π* = argmax_π V^π(s)

    Algorithms:
    - Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    - Policy Gradient: ∇_θ J = E[∇_θ log π_θ(a|s) Q^π(s,a)]
    - Actor-Critic: Combines both

Imitation Learning:
    Given expert demonstrations: {(s_i, a_i^expert)}
    Learn policy: π̂ ≈ π^expert

    Behavioral Cloning: Supervised learning
        L = Σ ||π̂(s_i) - a_i^expert||²

    Inverse RL: Learn reward function R̂
        Then solve RL with R̂

Usage
-----
>>> from src.types.learning import (
...     TrainingResult,
...     RLTrainingResult,
...     NeuralNetworkConfig,
... )
>>>
>>> # Train neural network dynamics model
>>> config: NeuralNetworkConfig = {
...     'hidden_layers': [64, 64, 32],
...     'activation': 'relu',
...     'learning_rate': 1e-3,
...     'batch_size': 32,
...     'epochs': 100,
... }
>>>
>>> result: TrainingResult = train_model(model, data, config)
>>> print(f"Final loss: {result['final_loss']:.3e}")
>>>
>>> # Reinforcement learning
>>> rl_result: RLTrainingResult = train_rl_agent(
...     env, algorithm='SAC', episodes=1000
... )
>>> policy = rl_result['learned_policy']
"""

from typing import Callable, Dict, List, Optional, Tuple

from typing_extensions import TypedDict

from .core import (
    ArrayLike,
    ControlVector,
    StateVector,
)
from .trajectories import (
    ControlSequence,
    OutputSequence,
    StateTrajectory,
)

# ============================================================================
# Type Aliases for Learning
# ============================================================================

Dataset = Tuple[StateTrajectory, ControlSequence, OutputSequence]
"""
System identification dataset: (states, controls, outputs).

Examples
--------
>>> states = np.random.randn(1000, 4)
>>> controls = np.random.randn(1000, 2)
>>> outputs = np.random.randn(1000, 2)
>>> dataset: Dataset = (states, controls, outputs)
"""

TrainingBatch = Tuple[StateVector, ControlVector, StateVector]
"""
Single training batch: (x[k], u[k], x[k+1]) for dynamics learning.

Examples
--------
>>> batch: TrainingBatch = (x_current, u_current, x_next)
"""

LearningRate = float
"""Learning rate for gradient-based optimization."""

LossValue = float
"""Loss/cost function value."""


# ============================================================================
# Neural Network Configuration
# ============================================================================


class NeuralNetworkConfig(TypedDict, total=False):
    """
    Neural network configuration for learning.

    Fields
    ------
    hidden_layers : List[int]
        Hidden layer sizes, e.g., [64, 64, 32]
    activation : str
        Activation function ('relu', 'tanh', 'sigmoid', 'elu')
    learning_rate : float
        Learning rate for optimizer
    batch_size : int
        Training batch size
    epochs : int
        Number of training epochs
    optimizer : str
        Optimizer type ('adam', 'sgd', 'rmsprop')
    regularization : Optional[float]
        L2 regularization strength
    dropout : Optional[float]
        Dropout probability (0-1)

    Examples
    --------
    >>> config: NeuralNetworkConfig = {
    ...     'hidden_layers': [128, 64, 32],
    ...     'activation': 'relu',
    ...     'learning_rate': 1e-3,
    ...     'batch_size': 64,
    ...     'epochs': 200,
    ...     'optimizer': 'adam',
    ...     'regularization': 1e-4,
    ...     'dropout': 0.1,
    ... }
    >>>
    >>> # Build model with config
    >>> model = build_neural_network(config)
    """

    hidden_layers: List[int]
    activation: str
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str
    regularization: Optional[float]
    dropout: Optional[float]


# ============================================================================
# Training Results
# ============================================================================


class TrainingResult(TypedDict, total=False):
    """
    Neural network training result.

    Fields
    ------
    final_loss : float
        Final training loss
    best_loss : float
        Best loss achieved during training
    loss_history : List[float]
        Loss value per epoch
    validation_loss : Optional[float]
        Validation set loss (if validation data provided)
    training_time : float
        Total training time in seconds
    epochs_trained : int
        Number of epochs completed
    early_stopped : bool
        Whether early stopping was triggered
    model_state : Optional[Dict]
        Saved model parameters/weights

    Examples
    --------
    >>> # Train neural ODE dynamics model
    >>> result: TrainingResult = train_neural_ode(
    ...     model, data, epochs=100, learning_rate=1e-3
    ... )
    >>>
    >>> print(f"Training complete:")
    >>> print(f"  Final loss: {result['final_loss']:.3e}")
    >>> print(f"  Best loss: {result['best_loss']:.3e}")
    >>> print(f"  Time: {result['training_time']:.1f}s")
    >>>
    >>> # Plot learning curve
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result['loss_history'])
    >>> plt.xlabel('Epoch')
    >>> plt.ylabel('Loss')
    >>> plt.yscale('log')
    >>>
    >>> if result['early_stopped']:
    ...     print("Training stopped early (convergence reached)")
    """

    final_loss: float
    best_loss: float
    loss_history: List[float]
    validation_loss: Optional[float]
    training_time: float
    epochs_trained: int
    early_stopped: bool
    model_state: Optional[Dict]


class NeuralDynamicsResult(TypedDict, total=False):
    """
    Learned neural network dynamics model result.

    Fields
    ------
    dynamics_model : Callable
        Learned f̂(x, u) → x_next
    prediction_error : float
        Mean prediction error on test set
    training_result : TrainingResult
        Training history and metrics
    architecture : List[int]
        Network architecture (layer sizes)

    Examples
    --------
    >>> # Learn dynamics from data
    >>> result: NeuralDynamicsResult = learn_dynamics(
    ...     states, controls, hidden_layers=[64, 64]
    ... )
    >>>
    >>> f_learned = result['dynamics_model']
    >>>
    >>> # Predict next state
    >>> x = np.array([1.0, 0.5])
    >>> u = np.array([0.1])
    >>> x_next_pred = f_learned(x, u)
    >>>
    >>> print(f"Prediction error: {result['prediction_error']:.3e}")
    >>> print(f"Architecture: {result['architecture']}")
    """

    dynamics_model: Callable
    prediction_error: float
    training_result: TrainingResult
    architecture: List[int]


# ============================================================================
# Reinforcement Learning
# ============================================================================


class RLTrainingResult(TypedDict, total=False):
    """
    Reinforcement learning training result.

    Fields
    ------
    learned_policy : Callable
        Trained policy π(s) → a
    episode_returns : List[float]
        Cumulative return per episode
    episode_lengths : List[int]
        Episode length (steps) per episode
    average_return : float
        Average return over last N episodes
    best_return : float
        Best episode return achieved
    total_timesteps : int
        Total environment steps
    training_time : float
        Training time in seconds
    converged : bool
        Whether training converged
    algorithm : str
        RL algorithm used ('DQN', 'PPO', 'SAC', 'TD3', etc.)

    Examples
    --------
    >>> # Train RL agent
    >>> result: RLTrainingResult = train_rl_agent(
    ...     env=pendulum_env,
    ...     algorithm='SAC',
    ...     episodes=1000,
    ...     learning_rate=3e-4
    ... )
    >>>
    >>> # Extract policy
    >>> policy = result['learned_policy']
    >>>
    >>> # Evaluate
    >>> print(f"Algorithm: {result['algorithm']}")
    >>> print(f"Average return: {result['average_return']:.2f}")
    >>> print(f"Best return: {result['best_return']:.2f}")
    >>>
    >>> # Plot learning curve
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result['episode_returns'])
    >>> plt.xlabel('Episode')
    >>> plt.ylabel('Return')
    >>> plt.title('RL Training Progress')
    >>>
    >>> # Deploy policy
    >>> state = env.reset()
    >>> action = policy(state)
    """

    learned_policy: Callable
    episode_returns: List[float]
    episode_lengths: List[int]
    average_return: float
    best_return: float
    total_timesteps: int
    training_time: float
    converged: bool
    algorithm: str


class PolicyEvaluationResult(TypedDict):
    """
    Policy evaluation result.

    Fields
    ------
    mean_return : float
        Mean cumulative return
    std_return : float
        Standard deviation of returns
    mean_episode_length : float
        Mean episode length
    success_rate : float
        Fraction of successful episodes (0-1)
    num_episodes : int
        Number of evaluation episodes

    Examples
    --------
    >>> # Evaluate trained policy
    >>> result: PolicyEvaluationResult = evaluate_policy(
    ...     policy, env, n_episodes=100
    ... )
    >>>
    >>> print(f"Performance:")
    >>> print(f"  Mean return: {result['mean_return']:.2f} ± {result['std_return']:.2f}")
    >>> print(f"  Success rate: {result['success_rate']*100:.1f}%")
    >>> print(f"  Avg episode length: {result['mean_episode_length']:.1f}")
    """

    mean_return: float
    std_return: float
    mean_episode_length: float
    success_rate: float
    num_episodes: int


# ============================================================================
# Imitation Learning
# ============================================================================


class ImitationLearningResult(TypedDict, total=False):
    """
    Imitation learning result.

    Learn from expert demonstrations.

    Fields
    ------
    learned_policy : Callable
        Learned policy π̂ ≈ π^expert
    imitation_error : float
        Mean squared error ||π̂(s) - π^expert(s)||²
    training_result : TrainingResult
        Training history
    num_demonstrations : int
        Number of expert demonstrations used
    method : str
        Method used ('behavioral_cloning', 'GAIL', 'DAgger')

    Examples
    --------
    >>> # Learn from expert demonstrations
    >>> expert_states = np.random.randn(500, 4)
    >>> expert_actions = np.random.randn(500, 2)
    >>>
    >>> result: ImitationLearningResult = imitation_learning(
    ...     expert_states, expert_actions,
    ...     method='behavioral_cloning',
    ...     epochs=100
    ... )
    >>>
    >>> policy = result['learned_policy']
    >>> print(f"Imitation error: {result['imitation_error']:.3e}")
    >>> print(f"Method: {result['method']}")
    >>> print(f"Demonstrations: {result['num_demonstrations']}")
    >>>
    >>> # Test learned policy
    >>> state = np.array([1.0, 0.5, 0.0, 0.0])
    >>> action = policy(state)
    """

    learned_policy: Callable
    imitation_error: float
    training_result: TrainingResult
    num_demonstrations: int
    method: str


# ============================================================================
# Online Adaptation
# ============================================================================


class OnlineAdaptationResult(TypedDict, total=False):
    """
    Online learning/adaptation result.

    For adaptive control with online parameter updates.

    Fields
    ------
    adapted_parameters : ArrayLike
        Current parameter estimates θ̂[k]
    parameter_history : List[ArrayLike]
        Parameter evolution over time
    adaptation_gains : ArrayLike
        Learning/adaptation rates
    tracking_error : float
        Current tracking error
    converged : bool
        Whether parameters converged
    adaptation_method : str
        Method used ('gradient', 'RLS', 'MRAC')

    Examples
    --------
    >>> # Online parameter adaptation
    >>> result: OnlineAdaptationResult = online_adapt(
    ...     system, reference_trajectory,
    ...     initial_params=theta0,
    ...     adaptation_rate=0.1
    ... )
    >>>
    >>> theta_final = result['adapted_parameters']
    >>> print(f"Final parameters: {theta_final}")
    >>> print(f"Tracking error: {result['tracking_error']:.3e}")
    >>>
    >>> # Plot parameter convergence
    >>> import matplotlib.pyplot as plt
    >>> param_history = np.array(result['parameter_history'])
    >>> plt.plot(param_history)
    >>> plt.xlabel('Time step')
    >>> plt.ylabel('Parameter value')
    >>> plt.legend([f'θ_{i}' for i in range(len(theta_final))])
    """

    adapted_parameters: ArrayLike
    parameter_history: List[ArrayLike]
    adaptation_gains: ArrayLike
    tracking_error: float
    converged: bool
    adaptation_method: str


# ============================================================================
# Export All
# ============================================================================

__all__ = [
    # Type aliases
    "Dataset",
    "TrainingBatch",
    "LearningRate",
    "LossValue",
    # Configuration
    "NeuralNetworkConfig",
    # Training results
    "TrainingResult",
    "NeuralDynamicsResult",
    # Reinforcement learning
    "RLTrainingResult",
    "PolicyEvaluationResult",
    # Imitation learning
    "ImitationLearningResult",
    # Online adaptation
    "OnlineAdaptationResult",
]
