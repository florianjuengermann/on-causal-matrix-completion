import numpy as np
import heapq
import networkx as nx
from networkx.algorithms.clique import find_cliques, find_cliques_recursive


def _biclique_find_(E, L, R, P, Q):
    bicliques = []
    P_disabled = set()
    for i in range(len(P)):
        _, x = P[i]
        if x in P_disabled:
            continue
        R_new = R.copy()
        R_new.append(x)
        L_new = [u for u in L if E[u][x]]
        C = [x]
        P_new, Q_new = [], []
        is_max = True
        for v in Q:
            N = [u for u in L_new if E[u][v]]
            if len(N) == len(L_new):
                is_max = False
                break
            elif len(N) > 0:
                Q_new.append(v)

        if is_max:
            for _, v in P:
                if v == x or v in P_disabled:
                    continue
                N = [u for u in L_new if E[u][v]]
                if len(N) == len(L_new):
                    R_new.append(v)
                    S = [u for u in L if not E[u][x] and E[u][v]]
                    if len(S) == 0:
                        C.append(v)
                elif len(N) > 0:
                    P_new.append((len(N), v))
            bicliques.append((L_new.copy(), R_new.copy()))
            if len(P_new) > 0:
                P_new = sorted([(c, x)
                                for c, x in P_new if not x in P_disabled])
                bicliques.extend(_biclique_find_(
                    E, L_new, R_new, P_new, Q_new))
        Q = Q + C
        for c in C:
            P_disabled.add(c)
    if len(bicliques) >= 100:
        print(f"returning {len(bicliques)}")
    return bicliques


def _biclique_find_opt(adj, L, R, P, Q):
    bicliques = []
    P_disabled = set()
    while len(P) > 0:
        _, x = heapq.heappop(P)
        if x in P_disabled:
            continue
        R_new = R.copy()
        R_new.append(x)
        L_new = L & adj[x]  # [u for u in L if E[u][x]]
        L_without_x = L - adj[x]
        C = [x]
        Q_new = []
        P_new = []
        is_max = True
        for v in Q:
            # N =   # [u for u in L_new if E[u][v]]
            if L_new <= adj[v]:
                is_max = False
                break
            elif not L_new.isdisjoint(adj[v]):
                Q_new.append(v)

        if is_max:
            for _, v in P:
                if v == x or v in P_disabled:
                    continue
                N = len(L_new & adj[v])  # [u for u in L_new if E[u][v]]
                if N == len(L_new):  # L_new <= adj[v]:
                    R_new.append(v)
                    # [u for u in L if not E[u][x] and E[u][v]]
                    #S = L_without_x & adj[v]
                    if L_without_x.isdisjoint(adj[v]):
                        C.append(v)
                elif N > 0:  # not L_new.isdisjoint(adj[v]):
                    P_new.append((N, v))
            bicliques.append((list(L_new), R_new.copy()))
            if len(P_new) > 0:
                #P_new = sorted(P_new)
                heapq.heapify(P_new)
                bicliques.extend(_biclique_find_opt(
                    adj, L_new, R_new, P_new, Q_new))
        Q.extend(C)
        for c in C:
            P_disabled.add(c)
    return bicliques


def _biclique_find_all(M):
    R_all = list(range(M.shape[0]))
    C_all = list(range(M.shape[1]))
    # first argument is only for the ordering
    P_init = [(np.sum(M[:, j]), j) for j in C_all]
    heapq.heapify(P_init)
    # P_init = sorted(P_init)  # sort to improve efficiency
    adj = [{u for u in R_all if M[u][v]} for v in C_all]
    bicliques = _biclique_find_opt(adj, set(R_all), [], P_init, [])
    return bicliques


def biclique_find(M, printStatus=False):
    '''
    Given a boolean adjacency matrix M,
    finds a maximal set of row and column indices R, C
    such that M[i][j] = True for all i∈R, j∈C.

    Uses the maximal biclique interation algorithm from
    Zhang et. al. '14:
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-110
    '''
    bicliques = list(_biclique_find_all_networkx(M))
    if printStatus:
        print(f"found {len(bicliques)} bicliques")
    smallerSize = [min(len(R), len(C)) for R, C in bicliques]
    bestInd = np.argmax(smallerSize)
    return bicliques[bestInd]


def anchor_sub_matrix(D, i, j, biclique_search=biclique_find, _retry=5):
    NC = np.where(D[i, :] == 1)[0]
    NR = np.where(D[:, j] == 1)[0]
    B = D[np.ix_(NR, NC)]
    ind_NR, ind_NC = biclique_search(B)
    ind_row, ind_col = NR[ind_NR], NC[ind_NC]

    if len(ind_row) == 0 or len(ind_col) == 0:
        if _retry <= 0:
            raise Exception("Did not find anchor matrix!")
        return anchor_sub_matrix(D, i, j, biclique_search=biclique_find, _retry=_retry-1)

    return ind_row, ind_col


def _biclique_find_all_networkx(B):
    (n_rows, n_cols) = B.shape
    A = np.block([[np.ones((n_rows, n_rows)), B],
                  [B.T, np.ones((n_cols, n_cols))]])
    G = nx.from_numpy_matrix(A)

    # find max clique that yields the most square (nxn) matrix
    cliques = find_cliques(G)
    #cliques = list(find_cliques_recursive(G))

    def genCliques():
        clique = next(cliques, None)
        while clique:
            clique = np.sort(clique)
            clique_rows_idx = clique[clique < n_rows]
            clique_cols_idx = clique[clique >= n_rows] - n_rows
            yield (clique_rows_idx, clique_cols_idx)
            clique = next(cliques, None)
    return genCliques()


def biclique_random(B):
    (n_rows, n_cols) = B.shape
    perm_row = np.random.permutation(np.arange(n_rows))
    perm_cols = np.random.permutation(np.arange(n_cols))
    #B_copy = B.copy()
    B_copy = B[np.ix_(perm_row, perm_cols)]
    cliques = _biclique_find_all_networkx(B_copy)
    ind_row, ind_col = next(cliques)
    return perm_row[ind_row], perm_cols[ind_col]


def whole_matrix(B):
    return np.arange(B.shape[0]), np.arange(B.shape[1])
