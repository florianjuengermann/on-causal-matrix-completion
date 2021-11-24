import numpy as np


def biclique_find_(E, L, R, P, Q):
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
                bicliques += biclique_find_(E, L_new, R_new, P_new, Q_new)
        Q = Q + C
        for c in C:
            P_disabled.add(c)
    return bicliques


def biclique_find_all(M):
    R_all = list(range(M.shape[0]))
    C_all = list(range(M.shape[1]))
    # first argument is only for the ordering
    P_init = [(np.sum(M[:, j]), j) for j in C_all]
    P_init = sorted(P_init)  # sort to improve efficiency
    bicliques = biclique_find_(M, R_all, [], P_init, [])
    return bicliques


def biclique_find(M, printStatus=False):
    ''' 
    Given a boolean adjacency matrix M,
    finds a maximal set of row and column indices R, C
    such that M[i][j] = True for all i∈R, j∈C.

    Uses the maximal biclique interation algorithm from
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-110
    '''
    bicliques = biclique_find_all(M)
    if printStatus:
        print(f"found {len(bicliques)} bicliques")
    smallerSize = [min(len(R), len(C)) for R, C in bicliques]
    bestInd = np.argmax(smallerSize)
    return bicliques[bestInd]


def anchor_sub_matrix(D, i, j):
    NC = np.where(D[i, :])[0]
    NR = np.where(D[:, j])[0]
    B = D[np.ix_(NR, NC)]
    ind_NR, ind_NC = biclique_find(B)
    ind_row, ind_col = NR[ind_NR], NC[ind_NC]

    assert(np.all(D[ind_row, j]))
    assert(np.all(D[i, ind_col]))
    assert(np.all(D[np.ix_(ind_row, ind_col)]))

    return ind_row, ind_col
