import numpy as np


def getRatingAndPropensityMatrix(inv_scale=1, alpha=0.01, r=5):
    num_users = int(80/inv_scale)  # m
    num_movies = int(80/inv_scale)  # n
    latent_dim = r
    num_core_users = int(20/inv_scale)  # m_core
    num_core_movies = int(20/inv_scale)  # n_core
    # users
    core_user_matrix = np.random.normal(size=(num_core_users, latent_dim))
    dirichlet_alphas = [alpha]*num_core_users
    B_transform_matrix = np.random.dirichlet(
        dirichlet_alphas, size=num_users-num_core_users)
    # is a linear combination of core users
    non_core_user_matrix = B_transform_matrix @ core_user_matrix
    latent_user_matrix = np.vstack((core_user_matrix, non_core_user_matrix))

    # movies
    core_movie_matrix = np.random.normal(size=(num_core_movies, latent_dim))
    dirichlet_alphas = [0.01]*num_core_movies
    B_transform_matrix = np.random.dirichlet(
        dirichlet_alphas, size=num_movies-num_core_movies)
    # is a linear combination of core movies
    non_core_movie_matrix = B_transform_matrix @ core_movie_matrix
    latent_movie_matrix = np.vstack((core_movie_matrix, non_core_movie_matrix))

    rating_matrix = latent_user_matrix @ latent_movie_matrix.T
    rating_var = np.var(rating_matrix)

    # scale to interval [1, 5]
    rating_matrix = rating_matrix * (5.0/(3*2*np.sqrt(rating_var))) + 3
    rating_matrix[rating_matrix > 5.0] = 5.0
    rating_matrix[rating_matrix < 1.0] = 1.0

    # propensity matrix P
    ind_core = np.ix_(range(num_core_users), range(
        num_core_movies))  # core both
    ind_users = np.ix_(range(num_core_users), range(
        num_core_movies, num_movies))  # core users, std movives
    ind_movies = np.ix_(range(num_core_users, num_users), range(
        num_core_movies))  # std users, core movies
    ind_std = np.ix_(range(num_core_users, num_users), range(
        num_core_movies, num_movies))  # std both

    high_low_threshold = 2.3
    rating_low_ind = rating_matrix <= high_low_threshold
    rating_high_ind = rating_matrix > high_low_threshold

    P = np.zeros((num_users, num_movies))

    data = [(ind_core, 0.7, 0.9), (ind_users, 0.35, 0.7),
            (ind_movies, 0.35, 0.7), (ind_std, 0.1, 0.05)]
    for ind, alpha, target_mean in data:
        mask = np.zeros_like(rating_matrix, dtype=bool)
        mask[ind] = 1
        P[mask & rating_low_ind] = np.power(
            alpha, rating_matrix[mask & rating_low_ind]-1)
        P[mask & rating_high_ind] = np.power(
            alpha, 5 - rating_matrix[mask & rating_high_ind])
        # scale in such a way that the mean is target_mean
        # but keep all values in [0, 1]:
        while abs(np.mean(P[mask]) - target_mean) > 1e-5:
            P[mask] *= 1/np.mean(P[mask]) * target_mean
            P[P > 1] = 1
            P[P < 0] = 0

    return rating_matrix, P


def getRatingAndPropensityMatrix_general(latent_movie_matrix=None, inv_scale=1, alpha=0.01, r=5):
    num_users = int(80/inv_scale)  # m
    num_movies = int(80/inv_scale)  # n
    latent_dim = r
    num_core_movies = int(30/inv_scale)  # n_core

    # users
    latent_user_matrix = np.random.normal(size=(num_users, latent_dim))

    # movies
    if latent_movie_matrix is None:
        core_movie_matrix = np.random.normal(
            size=(num_core_movies, latent_dim))
        dirichlet_alphas = [alpha]*num_core_movies
        B_transform_matrix = np.random.dirichlet(
            dirichlet_alphas, size=num_movies-num_core_movies)
        # is a linear combination of core movies
        non_core_movie_matrix = B_transform_matrix @ core_movie_matrix
        latent_movie_matrix = np.vstack(
            (core_movie_matrix, non_core_movie_matrix))

    rating_matrix = latent_user_matrix @ latent_movie_matrix.T
    rating_var = np.var(rating_matrix)

    # scale to interval [1, 5]
    rating_matrix = rating_matrix * (5.0/(3*2*np.sqrt(rating_var))) + 3
    rating_matrix[rating_matrix > 5.0] = 5.0
    rating_matrix[rating_matrix < 1.0] = 1.0

    # sparsity pattern:
    fav_genre = np.argmax(latent_user_matrix, axis=1)
    mov_genre = np.argmax(latent_movie_matrix, axis=1)

    ind_core_mov_all_user = np.ix_(range(num_users), range(num_core_movies))
    ind_non_core_mov_all_user = np.ix_(
        range(num_users), range(num_core_movies, num_movies))

    P = np.zeros((num_users, num_movies))
    P[ind_core_mov_all_user] = 1
    # only =1 if fav_genre and mov_genre agree
    P[ind_non_core_mov_all_user] = (np.outer(fav_genre, np.ones_like(mov_genre[num_core_movies:]))
                                    == np.outer(np.ones_like(fav_genre), mov_genre[num_core_movies:]))
    return rating_matrix, P, latent_movie_matrix
