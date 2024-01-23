import numpy as np

def sample_postive_test_samples(num_samples, min_num_dims=2, max_num_dims=5):

    num_dims = np.random.randint(min_num_dims, max_num_dims)
    mu = torch.randn((num_dims,))
    y = torch.randn((num_samples, num_dims)) + mu

    return y


