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
Unit Tests for Learning and Data-Driven Control Types

Tests TypedDict definitions and usage patterns for machine learning,
reinforcement learning, and adaptive control types.
"""

import pytest
import numpy as np
from src.types.learning import (
    Dataset,
    TrainingBatch,
    LearningRate,
    LossValue,
    NeuralNetworkConfig,
    TrainingResult,
    NeuralDynamicsResult,
    RLTrainingResult,
    PolicyEvaluationResult,
    ImitationLearningResult,
    OnlineAdaptationResult,
)


class TestTypeAliases:
    """Test type aliases for learning."""
    
    def test_dataset_structure(self):
        """Test dataset tuple structure."""
        states = np.random.randn(100, 4)
        controls = np.random.randn(100, 2)
        outputs = np.random.randn(100, 2)
        
        dataset: Dataset = (states, controls, outputs)
        
        x, u, y = dataset
        assert x.shape == (100, 4)
        assert u.shape == (100, 2)
        assert y.shape == (100, 2)
    
    def test_training_batch(self):
        """Test training batch structure."""
        batch: TrainingBatch = (
            np.array([1.0, 0.5]),      # x[k]
            np.array([0.1]),           # u[k]
            np.array([1.1, 0.55])      # x[k+1]
        )
        
        x_current, u_current, x_next = batch
        assert x_current.shape == (2,)
        assert u_current.shape == (1,)
        assert x_next.shape == (2,)


class TestNeuralNetworkConfig:
    """Test NeuralNetworkConfig TypedDict."""
    
    def test_config_creation(self):
        """Test creating network configuration."""
        config: NeuralNetworkConfig = {
            'hidden_layers': [128, 64, 32],
            'activation': 'relu',
            'learning_rate': 1e-3,
            'batch_size': 64,
            'epochs': 200,
            'optimizer': 'adam',
            'regularization': 1e-4,
            'dropout': 0.1,
        }
        
        assert len(config['hidden_layers']) == 3
        assert config['learning_rate'] > 0
        assert config['batch_size'] > 0
        assert config['epochs'] > 0
    
    def test_config_minimal(self):
        """Test config with minimal required fields."""
        config: NeuralNetworkConfig = {
            'hidden_layers': [64, 64],
            'activation': 'tanh',
            'learning_rate': 1e-4,
        }
        
        assert 'hidden_layers' in config
        assert 'activation' in config
    
    def test_config_different_activations(self):
        """Test different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid', 'elu']
        
        for act in activations:
            config: NeuralNetworkConfig = {
                'hidden_layers': [32],
                'activation': act,
                'learning_rate': 1e-3,
            }
            assert config['activation'] == act


class TestTrainingResult:
    """Test TrainingResult TypedDict."""
    
    def test_training_result_creation(self):
        """Test creating training result."""
        result: TrainingResult = {
            'final_loss': 0.001,
            'best_loss': 0.0008,
            'loss_history': [0.1, 0.05, 0.01, 0.005, 0.001],
            'validation_loss': 0.0012,
            'training_time': 45.3,
            'epochs_trained': 100,
            'early_stopped': False,
            'model_state': {'weights': 'saved'},
        }
        
        assert result['final_loss'] > 0
        assert result['best_loss'] <= result['final_loss']
        assert len(result['loss_history']) > 0
        assert result['epochs_trained'] > 0
    
    def test_training_convergence(self):
        """Test training convergence pattern."""
        # Decreasing loss
        loss_history = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
        
        result: TrainingResult = {
            'final_loss': loss_history[-1],
            'best_loss': min(loss_history),
            'loss_history': loss_history,
            'training_time': 30.0,
            'epochs_trained': len(loss_history),
            'early_stopped': False,
        }
        
        # Loss should generally decrease
        assert result['loss_history'][0] > result['loss_history'][-1]
        assert result['best_loss'] == min(loss_history)
    
    def test_training_early_stopping(self):
        """Test early stopping scenario."""
        result: TrainingResult = {
            'final_loss': 0.005,
            'best_loss': 0.005,
            'loss_history': [0.1, 0.05, 0.01, 0.005],
            'training_time': 20.0,
            'epochs_trained': 50,  # Stopped before 100
            'early_stopped': True,
        }
        
        assert result['early_stopped'] == True
        assert result['epochs_trained'] < 100  # Didn't complete


class TestNeuralDynamicsResult:
    """Test NeuralDynamicsResult TypedDict."""
    
    def test_neural_dynamics_creation(self):
        """Test creating neural dynamics result."""
        # Learned dynamics model
        def learned_f(x, u):
            return x + 0.1 * u  # Simple linear model
        
        training_result: TrainingResult = {
            'final_loss': 0.001,
            'best_loss': 0.001,
            'loss_history': [0.1, 0.01, 0.001],
            'training_time': 10.0,
            'epochs_trained': 50,
            'early_stopped': False,
        }
        
        result: NeuralDynamicsResult = {
            'dynamics_model': learned_f,
            'prediction_error': 0.005,
            'training_result': training_result,
            'architecture': [4, 64, 64, 4],
        }
        
        assert callable(result['dynamics_model'])
        assert result['prediction_error'] > 0
        assert len(result['architecture']) > 0
    
    def test_neural_dynamics_prediction(self):
        """Test using learned dynamics model."""
        # Simple learned model
        def f_learned(x, u):
            return x + 0.1 * np.concatenate([u, u])
        
        result: NeuralDynamicsResult = {
            'dynamics_model': f_learned,
            'prediction_error': 0.01,
            'training_result': {
                'final_loss': 0.01,
                'best_loss': 0.01,
                'loss_history': [0.1, 0.05, 0.01],
                'training_time': 5.0,
                'epochs_trained': 30,
                'early_stopped': False,
            },
            'architecture': [2, 32, 32, 2],
        }
        
        # Test prediction
        x = np.array([1.0, 0.5])
        u = np.array([0.1])
        x_next = result['dynamics_model'](x, u)
        
        assert x_next.shape == (2,)


class TestRLTrainingResult:
    """Test RLTrainingResult TypedDict."""
    
    def test_rl_training_result_creation(self):
        """Test creating RL training result."""
        policy = lambda s: np.array([0.1 * s[0]])
        
        result: RLTrainingResult = {
            'learned_policy': policy,
            'episode_returns': [10.0, 25.0, 50.0, 75.0, 90.0],
            'episode_lengths': [50, 60, 70, 80, 90],
            'average_return': 50.0,
            'best_return': 90.0,
            'total_timesteps': 350,
            'training_time': 120.0,
            'converged': True,
            'algorithm': 'SAC',
        }
        
        assert callable(result['learned_policy'])
        assert len(result['episode_returns']) > 0
        assert result['best_return'] >= result['average_return']
        assert result['total_timesteps'] > 0
    
    def test_rl_training_improvement(self):
        """Test RL training shows improvement."""
        # Returns should improve over time
        returns = [-100, -50, 0, 50, 100, 150, 180, 190, 195, 198]
        
        result: RLTrainingResult = {
            'learned_policy': lambda s: np.zeros(1),
            'episode_returns': returns,
            'episode_lengths': [100] * 10,
            'average_return': 91.3,  # Average of last 10
            'best_return': max(returns),
            'total_timesteps': 1000,
            'training_time': 60.0,
            'converged': True,
            'algorithm': 'PPO',
        }
        
        # Early returns should be lower than late returns
        early_avg = np.mean(returns[:3])
        late_avg = np.mean(returns[-3:])
        assert late_avg > early_avg
    
    def test_rl_different_algorithms(self):
        """Test different RL algorithms."""
        algorithms = ['DQN', 'PPO', 'SAC', 'TD3', 'DDPG']
        
        for alg in algorithms:
            result: RLTrainingResult = {
                'learned_policy': lambda s: np.zeros(1),
                'episode_returns': [10.0, 20.0, 30.0],
                'episode_lengths': [50, 50, 50],
                'average_return': 20.0,
                'best_return': 30.0,
                'total_timesteps': 150,
                'training_time': 10.0,
                'converged': False,
                'algorithm': alg,
            }
            assert result['algorithm'] == alg


class TestPolicyEvaluationResult:
    """Test PolicyEvaluationResult TypedDict."""
    
    def test_policy_evaluation_creation(self):
        """Test creating policy evaluation result."""
        result: PolicyEvaluationResult = {
            'mean_return': 85.5,
            'std_return': 12.3,
            'mean_episode_length': 78.2,
            'success_rate': 0.92,
            'num_episodes': 100,
        }
        
        assert result['mean_return'] > 0
        assert result['std_return'] >= 0
        assert 0 <= result['success_rate'] <= 1
        assert result['num_episodes'] > 0
    
    def test_policy_evaluation_statistics(self):
        """Test statistical properties."""
        # Simulate episode returns
        returns = np.random.randn(100) * 10 + 80
        
        result: PolicyEvaluationResult = {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'mean_episode_length': 75.0,
            'success_rate': 0.85,
            'num_episodes': len(returns),
        }
        
        # Standard deviation should be positive
        assert result['std_return'] > 0
        # Mean should be close to 80
        assert abs(result['mean_return'] - 80) < 5
    
    def test_policy_evaluation_success_rate(self):
        """Test success rate calculation."""
        # High performance policy
        result_good: PolicyEvaluationResult = {
            'mean_return': 150.0,
            'std_return': 10.0,
            'mean_episode_length': 100.0,
            'success_rate': 0.95,  # 95% success
            'num_episodes': 100,
        }
        
        # Poor performance policy
        result_poor: PolicyEvaluationResult = {
            'mean_return': 50.0,
            'std_return': 20.0,
            'mean_episode_length': 50.0,
            'success_rate': 0.30,  # 30% success
            'num_episodes': 100,
        }
        
        assert result_good['success_rate'] > result_poor['success_rate']


class TestImitationLearningResult:
    """Test ImitationLearningResult TypedDict."""
    
    def test_imitation_learning_creation(self):
        """Test creating imitation learning result."""
        policy = lambda s: 0.5 * s  # Linear policy
        
        result: ImitationLearningResult = {
            'learned_policy': policy,
            'imitation_error': 0.05,
            'training_result': {
                'final_loss': 0.05,
                'best_loss': 0.04,
                'loss_history': [0.5, 0.2, 0.1, 0.05],
                'training_time': 30.0,
                'epochs_trained': 50,
                'early_stopped': False,
            },
            'num_demonstrations': 500,
            'method': 'behavioral_cloning',
        }
        
        assert callable(result['learned_policy'])
        assert result['imitation_error'] >= 0
        assert result['num_demonstrations'] > 0
    
    def test_imitation_learning_methods(self):
        """Test different imitation learning methods."""
        methods = ['behavioral_cloning', 'GAIL', 'DAgger']
        
        for method in methods:
            result: ImitationLearningResult = {
                'learned_policy': lambda s: np.zeros(1),
                'imitation_error': 0.1,
                'training_result': {
                    'final_loss': 0.1,
                    'best_loss': 0.1,
                    'loss_history': [0.5, 0.3, 0.1],
                    'training_time': 20.0,
                    'epochs_trained': 30,
                    'early_stopped': False,
                },
                'num_demonstrations': 200,
                'method': method,
            }
            assert result['method'] == method
    
    def test_imitation_learning_quality(self):
        """Test imitation learning quality metrics."""
        # Good imitation (low error)
        result_good: ImitationLearningResult = {
            'learned_policy': lambda s: s,
            'imitation_error': 0.01,  # Very low
            'training_result': {
                'final_loss': 0.01,
                'best_loss': 0.01,
                'loss_history': [0.1, 0.05, 0.01],
                'training_time': 10.0,
                'epochs_trained': 50,
                'early_stopped': False,
            },
            'num_demonstrations': 1000,  # Many demos
            'method': 'behavioral_cloning',
        }
        
        # Poor imitation (high error)
        result_poor: ImitationLearningResult = {
            'learned_policy': lambda s: np.zeros_like(s),
            'imitation_error': 0.5,  # High error
            'training_result': {
                'final_loss': 0.5,
                'best_loss': 0.5,
                'loss_history': [1.0, 0.8, 0.5],
                'training_time': 5.0,
                'epochs_trained': 20,
                'early_stopped': False,
            },
            'num_demonstrations': 50,  # Few demos
            'method': 'behavioral_cloning',
        }
        
        assert result_good['imitation_error'] < result_poor['imitation_error']


class TestOnlineAdaptationResult:
    """Test OnlineAdaptationResult TypedDict."""
    
    def test_online_adaptation_creation(self):
        """Test creating online adaptation result."""
        result: OnlineAdaptationResult = {
            'adapted_parameters': np.array([2.5, 1.3, 0.8]),
            'parameter_history': [
                np.array([1.0, 1.0, 1.0]),
                np.array([1.5, 1.1, 0.9]),
                np.array([2.0, 1.2, 0.85]),
                np.array([2.5, 1.3, 0.8]),
            ],
            'adaptation_gains': np.array([0.1, 0.1, 0.1]),
            'tracking_error': 0.05,
            'converged': True,
            'adaptation_method': 'MRAC',
        }
        
        assert result['adapted_parameters'].shape == (3,)
        assert len(result['parameter_history']) > 0
        assert result['tracking_error'] >= 0
    
    def test_online_adaptation_convergence(self):
        """Test parameter convergence over time."""
        # Parameters should converge to true values
        true_params = np.array([2.0, 1.0, 0.5])
        
        # Simulate convergence
        history = [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.5, 1.0, 0.7]),
            np.array([1.8, 1.0, 0.6]),
            np.array([1.95, 1.0, 0.52]),
            np.array([2.0, 1.0, 0.5]),
        ]
        
        result: OnlineAdaptationResult = {
            'adapted_parameters': history[-1],
            'parameter_history': history,
            'adaptation_gains': np.array([0.1, 0.1, 0.1]),
            'tracking_error': 0.001,  # Very small
            'converged': True,
            'adaptation_method': 'gradient',
        }
        
        # Final parameters should be close to true
        assert np.allclose(result['adapted_parameters'], true_params, atol=0.05)
        assert result['converged'] == True
    
    def test_online_adaptation_methods(self):
        """Test different adaptation methods."""
        methods = ['gradient', 'RLS', 'MRAC']
        
        for method in methods:
            result: OnlineAdaptationResult = {
                'adapted_parameters': np.array([1.5, 0.8]),
                'parameter_history': [np.array([1.0, 1.0])],
                'adaptation_gains': np.array([0.05, 0.05]),
                'tracking_error': 0.1,
                'converged': False,
                'adaptation_method': method,
            }
            assert result['adaptation_method'] == method


class TestPracticalUseCases:
    """Test realistic usage patterns."""
    
    def test_neural_ode_training(self):
        """Test training neural ODE dynamics model."""
        # Generate synthetic data
        states = np.random.randn(500, 3)
        controls = np.random.randn(500, 1)
        next_states = states + 0.1 * np.tile(controls, (1, 3))
        
        # Train model
        result: NeuralDynamicsResult = {
            'dynamics_model': lambda x, u: x + 0.1 * np.tile(u, x.shape),
            'prediction_error': 0.01,
            'training_result': {
                'final_loss': 0.01,
                'best_loss': 0.01,
                'loss_history': [0.5, 0.1, 0.05, 0.01],
                'training_time': 45.0,
                'epochs_trained': 100,
                'early_stopped': False,
            },
            'architecture': [4, 64, 64, 3],
        }
        
        # Test prediction
        f = result['dynamics_model']
        x_test = np.array([1.0, 0.5, 0.0])
        u_test = np.array([0.1])
        x_next = f(x_test, u_test)
        
        assert x_next.shape == (3,)
    
    def test_rl_policy_deployment(self):
        """Test deploying trained RL policy."""
        # Trained policy
        def policy(state):
            # Simple proportional control
            return -0.5 * state
        
        result: RLTrainingResult = {
            'learned_policy': policy,
            'episode_returns': [i*10 for i in range(1, 11)],
            'episode_lengths': [100] * 10,
            'average_return': 55.0,
            'best_return': 100.0,
            'total_timesteps': 1000,
            'training_time': 180.0,
            'converged': True,
            'algorithm': 'SAC',
        }
        
        # Deploy policy
        state = np.array([1.0, 0.5])
        action = result['learned_policy'](state)
        
        assert action.shape == (2,)


class TestNumericalProperties:
    """Test numerical properties of results."""
    
    def test_loss_non_negative(self):
        """Test loss values are non-negative."""
        result: TrainingResult = {
            'final_loss': 0.001,
            'best_loss': 0.0005,
            'loss_history': [0.1, 0.05, 0.01, 0.005, 0.001],
            'training_time': 30.0,
            'epochs_trained': 50,
            'early_stopped': False,
        }
        
        assert result['final_loss'] >= 0
        assert result['best_loss'] >= 0
        assert all(loss >= 0 for loss in result['loss_history'])
    
    def test_learning_rate_positive(self):
        """Test learning rate is positive."""
        config: NeuralNetworkConfig = {
            'hidden_layers': [32],
            'activation': 'relu',
            'learning_rate': 1e-3,
        }
        
        assert config['learning_rate'] > 0
    
    def test_success_rate_bounds(self):
        """Test success rate is in [0, 1]."""
        result: PolicyEvaluationResult = {
            'mean_return': 75.0,
            'std_return': 10.0,
            'mean_episode_length': 80.0,
            'success_rate': 0.85,
            'num_episodes': 100,
        }
        
        assert 0 <= result['success_rate'] <= 1


class TestDocumentationExamples:
    """Test that documentation examples work."""
    
    def test_training_result_example(self):
        """Test TrainingResult example from docstring."""
        result: TrainingResult = {
            'final_loss': 0.001,
            'best_loss': 0.0008,
            'loss_history': [0.1, 0.05, 0.01, 0.001],
            'training_time': 30.0,
            'epochs_trained': 100,
            'early_stopped': False,
        }
        
        assert result['final_loss'] > 0
        assert len(result['loss_history']) > 0
    
    def test_rl_training_example(self):
        """Test RLTrainingResult example structure."""
        result: RLTrainingResult = {
            'learned_policy': lambda s: -0.5 * s,
            'episode_returns': [10, 20, 30, 40, 50],
            'episode_lengths': [50, 55, 60, 65, 70],
            'average_return': 30.0,
            'best_return': 50.0,
            'total_timesteps': 300,
            'training_time': 120.0,
            'converged': True,
            'algorithm': 'SAC',
        }
        
        assert callable(result['learned_policy'])
        assert result['algorithm'] == 'SAC'


class TestFieldPresence:
    """Test that all fields are accessible."""
    
    def test_training_result_has_required_fields(self):
        """Test TrainingResult has core fields."""
        result: TrainingResult = {
            'final_loss': 0.01,
            'best_loss': 0.01,
            'loss_history': [0.1, 0.05, 0.01],
            'training_time': 10.0,
            'epochs_trained': 30,
            'early_stopped': False,
        }
        
        assert 'final_loss' in result
        assert 'loss_history' in result
        assert 'epochs_trained' in result
    
    def test_rl_training_has_required_fields(self):
        """Test RLTrainingResult has core fields."""
        result: RLTrainingResult = {
            'learned_policy': lambda s: np.zeros(1),
            'episode_returns': [10.0],
            'episode_lengths': [50],
            'average_return': 10.0,
            'best_return': 10.0,
            'total_timesteps': 50,
            'training_time': 5.0,
            'converged': False,
            'algorithm': 'DQN',
        }
        
        assert 'learned_policy' in result
        assert 'episode_returns' in result
        assert 'algorithm' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])