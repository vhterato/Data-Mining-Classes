import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.semi_supervised import LabelPropagation
from sklearn.cluster import AffinityPropagation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import warnings

# Ignorar avisos para manter a saída limpa no console
warnings.filterwarnings('ignore')

def mapear_clusters_ap(cluster_labels, true_labels, labeled_indices):
    """
    Mapeia os clusters gerados pelo Affinity Propagation para as classes reais.
    Usa a classe majoritária dos dados rotulados que caíram em cada cluster.
    """
    cluster_to_class = {}
    
    # Classe majoritária global (usada caso um cluster não tenha nenhum dado rotulado)
    fallback_class = np.bincount(true_labels[labeled_indices]).argmax()
    
    for c in np.unique(cluster_labels):
        # Encontra quais exemplos rotulados pertencem a este cluster
        idx_in_cluster = [i for i in labeled_indices if cluster_labels[i] == c]
        
        if len(idx_in_cluster) > 0:
            # Atribui a classe mais frequente dentro do cluster
            classes_in_cluster = true_labels[idx_in_cluster]
            cluster_to_class[c] = np.bincount(classes_in_cluster).argmax()
        else:
            # Se o cluster estiver vazio de dados rotulados, usa o fallback global
            cluster_to_class[c] = fallback_class
            
    return cluster_to_class

def plotar_resultados(ratios_pct, historico):
    """
    Gera um gráfico com dois subplots (Acurácia e F1-Score) comparando o 
    desempenho de cada modelo ao longo de diferentes proporções de dados rotulados.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Marcadores e cores para facilitar a visualização de cada modelo
    modelos = {
        'Label Prop': ('lp', 'blue', 'o'),
        'Affinity Prop': ('ap', 'orange', 's'),
        'Logistic Reg': ('lr', 'green', '^'),
        'KNN': ('knn', 'red', 'D')
    }
    
    # Gráfico 1: Acurácia
    for nome, (chave, cor, marcador) in modelos.items():
        ax1.plot(ratios_pct, historico[f'{chave}_acc'], label=nome, color=cor, marker=marcador, linestyle='-')
    ax1.set_title('Acurácia vs % de Dados Rotulados')
    ax1.set_xlabel('% de Dados Rotulados')
    ax1.set_ylabel('Acurácia')
    ax1.set_xticks(ratios_pct)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Gráfico 2: F1-Score
    for nome, (chave, cor, marcador) in modelos.items():
        ax2.plot(ratios_pct, historico[f'{chave}_f1'], label=nome, color=cor, marker=marcador, linestyle='-')
    ax2.set_title('F1-Score (Macro) vs % de Dados Rotulados')
    ax2.set_xlabel('% de Dados Rotulados')
    ax2.set_ylabel('F1-Score')
    ax2.set_xticks(ratios_pct)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def avaliar_modelos(X, y):
    """
    Função principal que recebe matriz de características X e vetor de rótulos y,
    executa a avaliação comparativa e plota os resultados.
    """
    # Normalizar os dados é crucial para que os algoritmos baseados em distância
    # (LabelPropagation, KNN, AffinityPropagation) funcionem corretamente.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    ratios = np.arange(0.01, 0.11, 0.01) # Proporções de 1% a 10%
    n_splits = 5 # Número de divisões para a validação cruzada
    
    # Inicializando os classificadores base
    # Mudar o kernel para 'knn' corrige as falhas clássicas de propagação no kernel 'rbf'
    lp = LabelPropagation(kernel='knn', n_neighbors=7)
    ap = AffinityPropagation(random_state=42, damping=0.9, max_iter=200) 
    lr = LogisticRegression(max_iter=1000, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)

    print(f"{'Proporção (Rotulado/Não)':<25} | {'Label Prop (Acc|F1)':<20} | {'Affinity Prop (Acc|F1)':<22} | {'Logistic Reg (Acc|F1)':<21} | {'KNN (Acc|F1)':<15}")
    print("-" * 115)

    # Dicionário para armazenar o histórico e gerar os gráficos
    historico = {
        'ratios_pct': [],
        'lp_acc': [], 'lp_f1': [],
        'ap_acc': [], 'ap_f1': [],
        'lr_acc': [], 'lr_f1': [],
        'knn_acc': [], 'knn_f1': []
    }

    for ratio in ratios:
        metrics = {
            'lp': {'acc': [], 'f1': []},
            'ap': {'acc': [], 'f1': []},
            'lr': {'acc': [], 'f1': []},
            'knn': {'acc': [], 'f1': []}
        }
        
        # O StratifiedShuffleSplit garante o balanceamento das classes nos splits
        sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=ratio, random_state=42)
        
        for labeled_indices, unlabeled_indices in sss.split(X, y):
            # 1. Separar os dados rotulados e não rotulados
            X_labeled, y_labeled = X[labeled_indices], y[labeled_indices]
            X_unlabeled, y_unlabeled = X[unlabeled_indices], y[unlabeled_indices]
            
            # 2. Criar vetor de rótulos com -1 para os dados "não rotulados" (exigência do LabelProp)
            y_semi = np.copy(y)
            y_semi[unlabeled_indices] = -1
            
            # --- Treinamento e Predição ---
            
            # Label Propagation
            lp.fit(X, y_semi)
            lp_preds = lp.predict(X_unlabeled)
            
            # Affinity Propagation (não supervisionado - treinado em todo o X)
            ap_cluster_labels = ap.fit_predict(X)
            cluster_map = mapear_clusters_ap(ap_cluster_labels, y, labeled_indices)
            ap_preds = [cluster_map[c] for c in ap_cluster_labels[unlabeled_indices]]
            
            # Logistic Regression
            lr.fit(X_labeled, y_labeled)
            lr_preds = lr.predict(X_unlabeled)
            
            # K-Nearest Neighbors
            knn.fit(X_labeled, y_labeled)
            knn_preds = knn.predict(X_unlabeled)
            
            # --- Armazenamento das Métricas (usando 'macro' para lidar com classes desbalanceadas) ---
            metrics['lp']['acc'].append(accuracy_score(y_unlabeled, lp_preds))
            metrics['lp']['f1'].append(f1_score(y_unlabeled, lp_preds, average='macro'))
            
            metrics['ap']['acc'].append(accuracy_score(y_unlabeled, ap_preds))
            metrics['ap']['f1'].append(f1_score(y_unlabeled, ap_preds, average='macro'))
            
            metrics['lr']['acc'].append(accuracy_score(y_unlabeled, lr_preds))
            metrics['lr']['f1'].append(f1_score(y_unlabeled, lr_preds, average='macro'))
            
            metrics['knn']['acc'].append(accuracy_score(y_unlabeled, knn_preds))
            metrics['knn']['f1'].append(f1_score(y_unlabeled, knn_preds, average='macro'))

        # Calcular as médias para a proporção atual (após os n_splits)
        pct_labeled = int(round(ratio * 100))
        pct_unlabeled = 100 - pct_labeled
        ratio_str = f"{pct_labeled}% / {pct_unlabeled}%"
        
        lp_acc_avg, lp_f1_avg = np.mean(metrics['lp']['acc']), np.mean(metrics['lp']['f1'])
        ap_acc_avg, ap_f1_avg = np.mean(metrics['ap']['acc']), np.mean(metrics['ap']['f1'])
        lr_acc_avg, lr_f1_avg = np.mean(metrics['lr']['acc']), np.mean(metrics['lr']['f1'])
        knn_acc_avg, knn_f1_avg = np.mean(metrics['knn']['acc']), np.mean(metrics['knn']['f1'])
        
        # Salvar no histórico para o plot final
        historico['ratios_pct'].append(pct_labeled)
        historico['lp_acc'].append(lp_acc_avg); historico['lp_f1'].append(lp_f1_avg)
        historico['ap_acc'].append(ap_acc_avg); historico['ap_f1'].append(ap_f1_avg)
        historico['lr_acc'].append(lr_acc_avg); historico['lr_f1'].append(lr_f1_avg)
        historico['knn_acc'].append(knn_acc_avg); historico['knn_f1'].append(knn_f1_avg)
        
        # Exibir no terminal
        lp_str = f"{lp_acc_avg:.3f} | {lp_f1_avg:.3f}"
        ap_str = f"{ap_acc_avg:.3f} | {ap_f1_avg:.3f}"
        lr_str = f"{lr_acc_avg:.3f} | {lr_f1_avg:.3f}"
        knn_str = f"{knn_acc_avg:.3f} | {knn_f1_avg:.3f}"
        
        print(f"{ratio_str:<25} | {lp_str:<20} | {ap_str:<22} | {lr_str:<21} | {knn_str:<15}")

    # Chamar a função de plotagem com os dados compilados
    print("\nGerando os gráficos de desempenho...")
    plotar_resultados(historico['ratios_pct'], historico)

if __name__ == "__main__":
    # Exemplo de como usar a função com um dataset externo
    # Aqui, carregamos o dataset, fazemos um sub-sample para rodar rápido e passamos X e y
    """
    X_exemplo, y_exemplo = load_digits(return_X_y=True)
    
    # Reduzindo para as 1000 primeiras amostras apenas para o script rodar rápido no Affinity Propagation
    X_exemplo = X_exemplo[:1000]
    y_exemplo = y_exemplo[:1000]
    """

    X_exemplo = np.loadtxt("BCWD/xdata.dat")
    y_exemplo = np.loadtxt("BCWD/ydata.dat")
    y_exemplo = np.astype(y_exemplo,np.int64)

    # Reduzindo para as 1000 primeiras amostras apenas para o script rodar rápido no Affinity Propagation
    X_exemplo = X_exemplo[:1000]
    y_exemplo = y_exemplo[:1000]
    
    # Chamando a nova função modularizada
    avaliar_modelos(X_exemplo, y_exemplo)