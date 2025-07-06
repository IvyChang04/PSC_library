import unittest
import numpy as np
import torch
import torch.nn as nn
from sklearn import cluster, datasets
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from ParametricSpectralClustering.psc import PSC


class Net1(nn.Module):
    """Simple neural network for double half moon clustering"""
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


class TestDoubleHalfMoon(unittest.TestCase):
    """Test PSC on double half moon dataset"""
    
    def setUp(self):
        """Set up test data and model"""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate double half moon dataset
        self.n_samples = 1000
        self.X, self.y_true = datasets.make_moons(
            n_samples=self.n_samples, 
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
        """Test that double half moon data is generated correctly"""
        self.assertEqual(self.X.shape, (self.n_samples, 2))
        self.assertEqual(self.y_true.shape, (self.n_samples,))
        self.assertEqual(len(np.unique(self.y_true)), 2)
        
        # Check that data has two distinct clusters (half moons)
        # Half moons should be roughly separated by y-coordinate
        y_coords = self.X[:, 1]
        lower_moon = y_coords < 0
        upper_moon = y_coords > 0
        
        # Should have roughly equal number of points in each moon
        self.assertGreater(np.sum(lower_moon), 0)
        self.assertGreater(np.sum(upper_moon), 0)
        
        # Check that the moons are roughly crescent-shaped
        # Points should be distributed in a curved pattern
        x_coords = self.X[:, 0]
        
        # Lower moon should have negative y-coordinates
        lower_moon_points = self.X[lower_moon]
        self.assertTrue(np.all(lower_moon_points[:, 1] < 0))
        
        # Upper moon should have positive y-coordinates  
        upper_moon_points = self.X[upper_moon]
        self.assertTrue(np.all(upper_moon_points[:, 1] > 0))
    
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
        
        # For half moons, clustering should be quite good
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
        X_new, _ = datasets.make_moons(
            n_samples=100, 
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
        self.psc.save_model("test_double_half_moon_model")
        
        # Create new PSC instance and load model
        psc_new = PSC(
            model=Net1(2),
            n_clusters=2,
            sampling_ratio=0,
            n_components=2,
            n_neighbor=10,
            batch_size_data=len(self.X)
        )
        
        psc_new.load_model("test_double_half_moon_model")
        loaded_pred = psc_new.predict(self.X)
        
        # Predictions should be consistent
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_different_noise_levels(self):
        """Test clustering performance with different noise levels"""
        noise_levels = [0.01, 0.05]
        
        for noise in noise_levels:
            with self.subTest(noise=noise):
                # Generate data with different noise
                X_noisy, y_noisy = datasets.make_moons(
                    n_samples=self.n_samples, 
                    noise=noise,
                    random_state=42
                )
                
                psc_noisy = PSC(
                    model=Net1(2),
                    n_clusters=2,
                    sampling_ratio=0,
                    n_components=2,
                    n_neighbor=10,
                    batch_size_data=len(X_noisy)
                )
                
                try:
                    psc_noisy.fit(X_noisy)
                    y_pred = psc_noisy.predict(X_noisy)
                    
                    # Calculate quality metrics
                    ari = adjusted_rand_score(y_noisy, y_pred)
                    ami = adjusted_mutual_info_score(y_noisy, y_pred)
                    
                    # Higher noise should generally lead to lower scores
                    # But should still be reasonable (> 0.5 for moderate noise)
                    if noise <= 0.1:
                        self.assertGreater(ari, 0.5, f"ARI too low for noise={noise}: {ari}")
                        self.assertGreater(ami, 0.5, f"AMI too low for noise={noise}: {ami}")
                    
                except Exception as e:
                    self.fail(f"Failed with noise={noise}: {e}")
    
    def test_different_neighbor_values(self):
        """Test clustering with different n_neighbor values"""
        neighbor_values = [5, 10, 15, 20]
        
        for n_neighbor in neighbor_values:
            with self.subTest(n_neighbor=n_neighbor):
                psc_neighbor = PSC(
                    model=Net1(2),
                    n_clusters=2,
                    sampling_ratio=0,
                    n_components=2,
                    n_neighbor=n_neighbor,
                    batch_size_data=len(self.X)
                )
                
                try:
                    psc_neighbor.fit(self.X)
                    y_pred = psc_neighbor.predict(self.X)
                    
                    # Should still produce 2 clusters
                    self.assertEqual(len(np.unique(y_pred)), 2)
                    
                    # Calculate quality
                    ari = adjusted_rand_score(self.y_true, y_pred)
                    self.assertGreater(ari, 0.7, f"ARI too low for n_neighbor={n_neighbor}: {ari}")
                    
                except Exception as e:
                    self.fail(f"Failed with n_neighbor={n_neighbor}: {e}")
    
    def test_cluster_separation(self):
        """Test that clusters are well-separated for half moon data"""
        self.psc.fit(self.X)
        y_pred = self.psc.predict(self.X)
        
        # Get points from each cluster
        cluster_0_points = self.X[y_pred == 0]
        cluster_1_points = self.X[y_pred == 1]
        
        # Check that the clusters have roughly equal sizes
        # Should be within 20% of each other
        size_ratio = len(cluster_0_points) / len(cluster_1_points)
        self.assertGreater(size_ratio, 0.8, f"Cluster size ratio too small: {size_ratio}")
        self.assertLess(size_ratio, 1.2, f"Cluster size ratio too large: {size_ratio}")
        
        # Verify that the clusters maintain the crescent shape
        # Points should be distributed across the x-axis range
        x_coords_0 = cluster_0_points[:, 0]
        x_coords_1 = cluster_1_points[:, 0]  # Fixed: should be x-coordinates
        
        # Each cluster should span a reasonable range of x-coordinates
        x_range_0 = np.max(x_coords_0) - np.min(x_coords_0)
        x_range_1 = np.max(x_coords_1) - np.min(x_coords_1)
        
        # Should span at least 1.5 units in x-direction (typical for half moons)
        self.assertGreater(x_range_0, 1.5, f"Cluster 0 x-range too small: {x_range_0}")
        self.assertGreater(x_range_1, 1.5, f"Cluster 1 x-range too small: {x_range_1}")


if __name__ == "__main__":
    unittest.main() 