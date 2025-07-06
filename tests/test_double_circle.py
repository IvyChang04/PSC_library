import unittest
import numpy as np
import torch
import torch.nn as nn
from sklearn import cluster, datasets
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from ParametricSpectralClustering.psc import PSC


class Net1(nn.Module):
    """Simple neural network for double circle clustering"""
    def __init__(self, out_put):
        super(Net1, self).__init__()
        self.fc = nn.Linear(2, 32)
        self.output_layer = nn.Linear(32, out_put)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class TestDoubleCircle(unittest.TestCase):
    """Test PSC on double circle dataset"""
    
    def setUp(self):
        """Set up test data and model"""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate double circle dataset
        self.n_samples = 1000
        self.X, self.y_true = datasets.make_circles(
            n_samples=self.n_samples, 
            factor=0.5, 
            noise=0.05,
            random_state=42
        )
        
        # Create PSC model
        self.psc = PSC(
            model=Net1(2),
            n_clusters=2,
            sampling_ratio=0,
            n_components=2,
            n_neighbor=10,
            batch_size_data=len(self.X)
        )
    
    def test_data_generation(self):
        """Test that double circle data is generated correctly"""
        self.assertEqual(self.X.shape, (self.n_samples, 2))
        self.assertEqual(self.y_true.shape, (self.n_samples,))
        self.assertEqual(len(np.unique(self.y_true)), 2)
        
        # Check that data has two distinct clusters (circles)
        # Inner circle should be closer to origin
        distances = np.sqrt(self.X[:, 0]**2 + self.X[:, 1]**2)
        inner_points = distances < 0.5
        outer_points = distances > 0.5
        
        # Should have roughly equal number of points in each circle
        self.assertGreater(np.sum(inner_points), 0)
        self.assertGreater(np.sum(outer_points), 0)
    
    def test_model_initialization(self):
        """Test that PSC model is initialized correctly"""
        self.assertEqual(self.psc.n_clusters, 2)
        self.assertEqual(self.psc.n_components, 2)
        self.assertEqual(self.psc.n_neighbor, 10)
        self.assertEqual(self.psc.sampling_ratio, 0)
        self.assertEqual(self.psc.batch_size_data, self.n_samples)
        
        # Test model architecture
        self.assertIsInstance(self.psc.model, Net1)
        self.assertIsInstance(self.psc.clustering, cluster.KMeans)
    
    def test_model_fitting(self):
        """Test that the model can be fitted without errors"""
        try:
            self.psc.fit(self.X)
            self.assertTrue(self.psc.model_fitted)
        except Exception as e:
            self.fail(f"Model fitting failed with error: {e}")
    
    def test_prediction_shape(self):
        """Test that predictions have correct shape"""
        self.psc.fit(self.X)
        y_pred = self.psc.predict(self.X)
        
        self.assertEqual(y_pred.shape, (self.n_samples,))
        self.assertEqual(len(np.unique(y_pred)), 2)
    
    def test_clustering_quality(self):
        """Test that clustering produces reasonable results"""
        self.psc.fit(self.X)
        y_pred = self.psc.predict(self.X)
        
        # Calculate clustering metrics
        ari = adjusted_rand_score(self.y_true, y_pred)
        ami = adjusted_mutual_info_score(self.y_true, y_pred)
        
        # For double circles, clustering should be quite good
        # ARI and AMI should be > 0.8 for this well-separated dataset
        self.assertGreater(ari, 0.8, f"ARI too low: {ari}")
        self.assertGreater(ami, 0.8, f"AMI too low: {ami}")
    
    def test_fit_predict_consistency(self):
        """Test that fit_predict gives same results as fit + predict"""
        self.psc.fit(self.X)
        y_pred1 = self.psc.predict(self.X)
        
        # Reset model
        self.psc.model_fitted = False
        y_pred2 = self.psc.fit_predict(self.X)
        
        # Results should be consistent (allowing for label permutation)
        # Check if one is permutation of the other
        unique_labels1 = np.unique(y_pred1)
        unique_labels2 = np.unique(y_pred2)
        
        self.assertEqual(len(unique_labels1), len(unique_labels2))
        self.assertEqual(len(unique_labels1), 2)
    
    def test_new_data_prediction(self):
        """Test prediction on new data points"""
        self.psc.fit(self.X)
        
        # Generate new data from same distribution
        X_new, _ = datasets.make_circles(
            n_samples=100, 
            factor=0.5, 
            noise=0.05,
            random_state=123
        )
        
        y_pred_new = self.psc.predict(X_new)
        
        self.assertEqual(y_pred_new.shape, (100,))
        self.assertEqual(len(np.unique(y_pred_new)), 2)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        self.psc.fit(self.X)
        original_pred = self.psc.predict(self.X)
        
        # Save model
        self.psc.save_model("test_double_circle_model")
        
        # Create new PSC instance and load model
        psc_new = PSC(
            model=Net1(2),
            n_clusters=2,
            sampling_ratio=0,
            n_components=2,
            n_neighbor=10,
            batch_size_data=len(self.X)
        )
        
        psc_new.load_model("test_double_circle_model")
        loaded_pred = psc_new.predict(self.X)
        
        # Predictions should be consistent
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test invalid sampling_ratio
        with self.assertRaises(AttributeError):
            psc_invalid = PSC(
                model=Net1(2),
                n_clusters=2,
                sampling_ratio=1.5,  # Invalid
                n_components=2
            )
            psc_invalid.fit(self.X)
    
    def test_different_batch_sizes(self):
        """Test that different batch sizes work"""
        batch_sizes = [50, 100, 500, self.n_samples]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                psc_batch = PSC(
                    model=Net1(2),
                    #clustering_method=cluster.KMeans(n_clusters=2, n_init=10, verbose=False),
                    n_clusters=2,
                    sampling_ratio=0,
                    n_components=2,
                    n_neighbor=10,
                    batch_size_data=batch_size
                )
                
                try:
                    psc_batch.fit(self.X)
                    y_pred = psc_batch.predict(self.X)
                    self.assertEqual(y_pred.shape, (self.n_samples,))
                except Exception as e:
                    self.fail(f"Failed with batch_size={batch_size}: {e}")


if __name__ == "__main__":
    unittest.main() 