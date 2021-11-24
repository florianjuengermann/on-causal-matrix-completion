from snn import SyntheticNearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import generator as gen
import ridgeNN
# Construct the test data


params = {
    'n_neighbors': 1,
    'weights': 'distance',
    'verbose': False,
    # 'spectral_t': 0.99,
    'max_rank': 8,
    'min_value': 1,
    'max_value': 5,
}
snn = SyntheticNearestNeighbors(**params)

_, _, latent_movie_matrix = gen.getRatingAndPropensityMatrix_general()
RMSEs = []
MAEs = []
for _ in range(10):
    # pass the old latent_movie_matrix so we do not recompute it
    # only recompute rating_matrix
    rating_matrix, P, latent_movie_matrix = gen.getRatingAndPropensityMatrix_general(
        latent_movie_matrix)
    D = np.random.binomial(1, P)  # not really needed as P[i,j] ∈ {0, 1}
    Y = rating_matrix * D
    Y[D == 0] = np.nan
    Y_restored = snn.fit_transform(Y)
    #Y_restored = ridgeNN.RidgeNN(D, Y)
    Error = (rating_matrix - Y_restored).flatten()
    RMSEs.append(np.sqrt(np.mean(Error ** 2)))
    MAEs.append(np.mean(np.abs(Error)))

print(f"RMSE: {np.mean(RMSEs):.4f}±{np.sqrt(np.var(RMSEs)):.3f}")
print(f"MAE: {np.mean(MAEs):.4f}±{np.sqrt(np.var(MAEs)):.3f}")
# distribution of true vs restored samples:
plt.hist(rating_matrix.flatten(), bins=100, alpha=0.3)
plt.hist(Y_restored.flatten(), bins=100, color='C0', alpha=0.7)
plt.xlabel('ratings')
plt.ylabel('frequency')
# plt.show()
