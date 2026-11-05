import numpy as np

def jwSimilarity(dist, knn):
    n = len(dist)
    sigma = 0
    edges = []
    for i in range(n):
        aux = np.argsort(dist[i])
        aux_max = 0
        for k in range(1,knn+1):
            if [aux[k],i] not in edges:
                edges.append([i,aux[k]])
                if dist[i][aux[k]]>aux_max:
                    aux_max = dist[i][aux[k]]
        sigma += np.sqrt(aux_max) 
    sigma /= 3*n

    for e in edges:
        e.append(np.exp(-(dist[e[0]][e[1]]**2)/(2*sigma*sigma)))
    return edges