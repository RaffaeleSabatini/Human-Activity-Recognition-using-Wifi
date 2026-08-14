import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_dataset(cols, rows, dataset, activities, labels):
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    axes = axes.flatten()  

    for i in range(cols * rows):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label_idx = dataset[sample_idx]
        
        ax = axes[i]
        im = ax.imshow(img[0, :, :], aspect='auto', cmap='viridis')
        
        ax.set_title(f"{activities[label_idx]} ({labels[label_idx]})", fontsize=10, pad=8)
        
        ax.set_xlabel("Time (bin)", fontsize=8)
        ax.set_ylabel(r"$v_p \cos \alpha$ (bin)", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Power (db)', fontsize=8)

    plt.tight_layout()
    plt.show()



def plot_loss(train_loss, test_loss, train_accuracy, test_accuracy, title=""):
    epochs = range(1, len(train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title)

    # --------------------------------------------------------------------------
    # GRAPH 1: LOSS HISTORY
    # --------------------------------------------------------------------------
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, test_loss, 'r-s', label='Test Loss (Fusion)', linewidth=2, markersize=4)
    
    min_test_loss_idx = np.argmin(test_loss)
    ax1.axvline(x=min_test_loss_idx + 1, color='gray', linestyle='--', alpha=0.7, 
                label=f'Best Epoch ({min_test_loss_idx + 1})')
    ax1.plot(min_test_loss_idx + 1, test_loss[min_test_loss_idx], 'go', markersize=10, 
             label=f'Min Test Loss: {test_loss[min_test_loss_idx]:.4f}')
    
    ax1.set_title('Andamento della Loss durante l\'Addestramento', fontsize=12, pad=10)
    ax1.set_xlabel('Epoca', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9, loc='upper right')
    
    # --------------------------------------------------------------------------
    # GRAPH 2: ACCURACY HISTORY
    # --------------------------------------------------------------------------
    ax2.plot(epochs, train_accuracy, 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, test_accuracy, 'r-s', label='Test Accuracy (Fusion)', linewidth=2, markersize=4)
    
    # Evidenzia il punto di massima accuratezza nel Test
    max_test_acc_idx = np.argmax(test_accuracy)
    ax2.plot(max_test_acc_idx + 1, test_accuracy[max_test_acc_idx], 'go', markersize=10, 
             label=f'Max Test Accuracy: {test_accuracy[max_test_acc_idx]:.1f}%')
    
    ax2.set_title('Andamento dell\'Accuratezza', fontsize=12, pad=10)
    ax2.set_xlabel('Epoca', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crea le annotazioni di testo combinando il conteggio assoluto e la percentuale
    labels = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            abs_val = cm[i, j]
            perc_val = cm_normalized[i, j] * 100
            # Se la cella è vuota metti solo 0, altrimenti metti valore e percentuale
            if abs_val > 0:
                row.append(f"{abs_val:.2f}\n({perc_val:.1f}%)")
            else:
                row.append("0")
        labels.append(row)
    labels = np.asarray(labels)
    
    # Inizializza la figura
    plt.figure(figsize=(10, 8))
    
    # Plotta la heatmap
    # Usiamo il cmap 'Blues' o 'YlGnBu' che rende molto leggibili i blocchi sulla diagonale
    sns.heatmap(
        cm_normalized, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        cbar=True,
        xticklabels=class_names, 
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        annot_kws={"size": 11, "weight": "bold"} # Font dei numeri dentro le celle
    )
    
    # Label e formattazione assi
    plt.title("Matrice di Confusione del Modello", fontsize=14, pad=15, weight='bold')
    plt.ylabel("Classe Reale (Ground Truth)", fontsize=12, labelpad=10)
    plt.xlabel("Classe Predetta", fontsize=12, labelpad=10)
    
    # Ruota i tick per evitare che si sovrappongano se i nomi sono lunghi
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_f1_score(precisions, recalls, f1_scores, class_names):
    data = np.array([precisions, recalls, f1_scores]).T
    metric_names = ["Precision", "Recall", "F1-Score"]
    
    # Prepariamo i testi da mostrare dentro ogni cella (formato percentuale)
    annot_labels = np.array([[f"{val:.2%}" for val in row] for row in data])

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        data,
        annot=annot_labels,
        fmt="",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Punteggio (0 - 1)'},
        xticklabels=metric_names,
        yticklabels=class_names,
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=1.5,  # Separazione netta tra le celle
        linecolor="white"
    )
    
    # Label e formattazione
    plt.title("Report delle Metriche per Classe", fontsize=14, pad=18, weight='bold')
    plt.xlabel("Metriche di Performance", fontsize=11, labelpad=12, weight='semibold')
    plt.ylabel("Classi", fontsize=11, labelpad=12, weight='semibold')
    
    # Ruotiamo leggermente i tick 
    plt.xticks(fontsize=10, weight='semibold')
    plt.yticks(rotation=0, fontsize=10, weight='semibold')
    
    plt.tight_layout()
    plt.show()
