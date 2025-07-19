import sys
import pickle
import cloudpickle

# Monkey patch to handle missing sklearn modules
class MockModule:
    def __getattr__(self, name):
        return MockModule()

sys.modules['sklearn.ensemble._gb_losses'] = MockModule()

def load_legacy_model(filepath):
    """Load a scikit-learn model saved with an older version."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        # Try with cloudpickle
        with open(filepath, 'rb') as f:
            return cloudpickle.load(f)

if __name__ == "__main__":
    # Test loading the model
    model = load_legacy_model('phish_model.pkl')
    print(f"Model loaded successfully: {type(model)}")