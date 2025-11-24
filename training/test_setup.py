"""
Quick setup verification script
Tests that all components are working correctly
"""

import sys
import os

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        from kaggle_environments import make
        print(f"✓ Kaggle Environments")
    except ImportError as e:
        print(f"✗ Kaggle Environments import failed: {e}")
        return False
    
    return True


def test_modules():
    """Test that all custom modules can be imported."""
    print("\nTesting custom modules...")
    
    modules = [
        'config',
        'utils',
        'replay_buffer',
        'dqn_model',
        'dqn_agent',
        'train_dqn',
        'visualize',
        'connectX_Agent'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py")
        except ImportError as e:
            print(f"✗ {module}.py failed: {e}")
            return False
    
    return True


def test_model_creation():
    """Test that model can be created."""
    print("\nTesting model creation...")
    
    try:
        from dqn_model import create_model, count_parameters
        from config import config
        
        model = create_model('standard')
        params = count_parameters(model)
        
        print(f"✓ Model created successfully")
        print(f"  Device: {config.DEVICE}")
        print(f"  Parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_agent_creation():
    """Test that agent can be created."""
    print("\nTesting agent creation...")
    
    try:
        from dqn_agent import DQNAgent
        
        agent = DQNAgent(model_type='standard', use_double_dqn=True)
        
        print(f"✓ Agent created successfully")
        print(f"  Epsilon: {agent.epsilon}")
        print(f"  Buffer size: {len(agent.memory)}")
        
        return True
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False


def test_state_encoding():
    """Test state encoding."""
    print("\nTesting state encoding...")
    
    try:
        from utils import encode_state
        from config import config
        
        # Empty board
        board = [0] * (config.ROWS * config.COLUMNS)
        state = encode_state(board, mark=1)
        
        expected_shape = (3, config.ROWS, config.COLUMNS)
        assert state.shape == expected_shape, f"Expected {expected_shape}, got {state.shape}"
        
        print(f"✓ State encoding works")
        print(f"  Shape: {state.shape}")
        
        return True
    except Exception as e:
        print(f"✗ State encoding failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("DQN ConnectX Setup Verification")
    print("="*60)
    
    tests = [
        test_imports,
        test_modules,
        test_model_creation,
        test_agent_creation,
        test_state_encoding
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append(False)
        print()
    
    print("="*60)
    if all(results):
        print("✓ All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Run 'python train_dqn.py' to train the agent")
        print("2. Run 'python connectX_Agent.py' to test the trained agent")
        print("="*60)
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nMake sure you have installed all requirements:")
        print("  pip install -r requirements.txt")
        print("="*60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

