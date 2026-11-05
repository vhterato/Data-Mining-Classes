import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import warnings

# Ignorar avisos de desconexão de grafos para vizinhanças muito pequenas
warnings.filterwarnings('ignore')

def plotar_reducao_dimensional(X, y, vizinhos=[5, 15, 50]):
    """
    Reduz a dimensionalidade de X para 2D usando PCA e Laplacian Eigenmaps
    (Spectral Embedding) e plota os resultados comparativos coloridos por y.
    
    Parâmetros:
    X (array-like): Matriz de características
    y (array-like): Vetor de rótulos
    vizinhos (list): Lista com três valores inteiros para n_neighbors no Laplacian Eigenmaps
    """
    if len(vizinhos) != 3:
        raise ValueError("A lista de 'vizinhos' deve conter exatamente 3 valores.")

    # 1. Normalizar os dados (Crítico para PCA e algoritmos baseados em distância/grafos)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Criar a figura para os 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    # 2. PCA (Redução Linear)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot do PCA
    scatter_pca = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=40, alpha=0.8)
    axes[0].set_title('PCA (Linear)')
    axes[0].set_xlabel('Componente Principal 1')
    axes[0].set_ylabel('Componente Principal 2')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 3. Laplacian Eigenmaps (Spectral Embedding - Não linear) para diferentes vizinhos
    for i, n_neighbors in enumerate(vizinhos):
        # A afinidade 'nearest_neighbors' constrói o grafo de vizinhança característico do Laplacian Eigenmaps
        se = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', 
                               n_neighbors=n_neighbors, random_state=42)
        X_se = se.fit_transform(X_scaled)
        
        # Plot do Laplacian Eigenmaps
        axes[i+1].scatter(X_se[:, 0], X_se[:, 1], c=y, cmap='viridis', edgecolor='k', s=40, alpha=0.8)
        axes[i+1].set_title(f'Laplacian Eigenmaps\n(n_neighbors={n_neighbors})')
        axes[i+1].set_xlabel('Dimensão Embutida 1')
        axes[i+1].set_ylabel('Dimensão Embutida 2')
        axes[i+1].grid(True, linestyle='--', alpha=0.5)
    
    # Adicionando uma barra de cores global caso as classes sejam contínuas ou muitas
    cbar = fig.colorbar(scatter_pca, ax=axes.tolist(), orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Rótulos (y)')
    
    plt.suptitle('Comparação de Redução Dimensional para 2D: PCA vs Laplacian Eigenmaps', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96]) # Ajuste para caber a suptitle e a colorbar
    plt.show()

if __name__ == "__main__":
    # Gerando um dataset sintético não-linear (Moons) com ruído 
    # para demonstrar a superioridade de métodos baseados em grafos (Manifold)
    print("Gerando dataset de exemplo e processando a redução dimensional...")
    #X_exemplo, y_exemplo = make_moons(n_samples=500, noise=0.1, random_state=42)

    X_exemplo = np.loadtxt("iris/xdata.dat")
    y_exemplo = np.loadtxt("iris/ydata.dat")
    y_exemplo = np.astype(y_exemplo,np.int64)
    
    # Aqui chamamos a função principal com os dados
    plotar_reducao_dimensional(X_exemplo, y_exemplo, vizinhos=[5, 15, 50])