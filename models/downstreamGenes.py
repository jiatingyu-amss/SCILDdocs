import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import copy

# ---------- Dataset ----------
class CommunicationDataset(Dataset):
    def __init__(self, T_receiver, Y):
        self.X = torch.tensor(T_receiver, dtype=torch.float32).view(T_receiver.shape[0], -1)  # flatten (L*R)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ---------- Model ----------
class CommToGeneModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CommToGeneModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*4),
            nn.BatchNorm1d(hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)
        )
        
    
    def forward(self, x):
        return self.model(x)


# ---------- Train Function ----------
def train_model(
    T_receiver,
    Y,
    hidden_dim=64,
    epochs=200,
    batch_size=64,
    lr=1e-3,
    val_split=0.2,
    patience=10,
    min_delta=1e-4,
    plot_loss=True
):
    from tqdm import tqdm
    dataset = CommunicationDataset(T_receiver, Y)

    # Split into train and validation
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = T_receiver.shape[1] * T_receiver.shape[2]  # L * R
    output_dim = Y.shape[1]  # num_genes

    model = CommToGeneModel(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    with tqdm(total=epochs, desc="Training Epochs") as pbar:
        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, Y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    output = model(X_batch)
                    loss = criterion(output, Y_batch)
                    total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'Epoch': epoch,
                'Train Loss': f'{avg_train_loss:.4f}',
                'Val Loss': f'{avg_val_loss:.4f}',
                'LR': f'{current_lr:.6f}'
            })

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

            # Early stopping check
            if best_val_loss - avg_val_loss > min_delta:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            pbar.update(1)

    model.train_losses = train_losses
    model.val_losses = val_losses

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if plot_loss:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return model


def ablation_analysis(model, T_receiver, l_idx, r_idx, 
                      gene_names=None, top_k=10, title_=None, save_file=None):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    """
    Args:
    model: Trained model for gene expression prediction
    T_receiver: Receiver cell tensor (input to model)
    l_idx: Index of ligand to ablate
    r_idx: Index of receptor to ablate
    gene_names: Optional list of gene names for visualization
    top_k: Number of top affected genes to display (by magnitude)
    title_: Optional title for the plot
    """

    model.eval()

    # Original input
    T_original = torch.tensor(T_receiver, dtype=torch.float32).view(T_receiver.shape[0], -1)

    # Ablated input: Create copy with specific l-r channel zeroed out
    T_ablate = T_receiver.copy()
    T_ablate[:, l_idx, r_idx] = 0.0  # Knockout this communication pair
    T_ablate = torch.tensor(T_ablate, dtype=torch.float32).view(T_receiver.shape[0], -1)

    # Predict expression before and after ablation
    with torch.no_grad():
        Y_pred_orig = model(T_original)
        Y_pred_ablate = model(T_ablate)

    # Calculate per-gene average expression change (directional)
    delta = (Y_pred_ablate-Y_pred_orig).numpy()  # [num_cells, num_genes]
    mean_delta = np.mean(delta, axis=0)  # signed change
    abs_delta = np.abs(mean_delta)

    # Top-k most affected genes by absolute change
    top_k_idx = np.argsort(abs_delta)[::-1][:top_k]

    print(f"\nTop {top_k} affected genes (by magnitude) after ablating L={l_idx}, R={r_idx}:")
    for i, gene_i in enumerate(top_k_idx):
        gene_name = gene_names[gene_i] if gene_names is not None and gene_i < len(gene_names) else f"Gene{gene_i}"
        direction = "↑" if mean_delta[gene_i] > 0 else "↓"
        print(f"{i+1}. {gene_name} — Δexpr = {mean_delta[gene_i]:+.4f} {direction}")

    # Top 5 upregulated
    top_up_idx = np.argsort(mean_delta)[::-1][:5]
    print("\nTop 5 Upregulated Genes:")
    for i, gene_i in enumerate(top_up_idx):
        gene_name = gene_names[gene_i] if gene_names is not None and gene_i < len(gene_names) else f"Gene{gene_i}"
        print(f"{i+1}. {gene_name} — Δexpr = {mean_delta[gene_i]:+.4f} ↑")

    # Top 5 downregulated
    top_down_idx = np.argsort(mean_delta)[:5]
    print("\nTop 5 Downregulated Genes:")
    for i, gene_i in enumerate(top_down_idx):
        gene_name = gene_names[gene_i] if gene_names is not None and gene_i < len(gene_names) else f"Gene{gene_i}"
        print(f"{i+1}. {gene_name} — Δexpr = {mean_delta[gene_i]:+.4f} ↓")

    # # Visualization (top_k)
    # plt.figure(figsize=(8, 4))
    # bar_labels = [gene_names[i] if gene_names and i < len(gene_names) else f"G{i}" for i in top_k_idx]
    # plt.bar(bar_labels, mean_delta[top_k_idx], color=['red' if mean_delta[i] < 0 else 'green' for i in top_k_idx])
    # plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    # plt.xticks(rotation=45)
    # plt.ylabel("Mean Δ expression (directional)")
    # plt.title(title_ or f"Ablation impact of L={l_idx}, R={r_idx}")
    # plt.tight_layout()
    # plt.show()

    # # Visualization: compare original vs ablated prediction for top_k genes
    # plt.figure(figsize=(10, 5))
    # indices = np.arange(top_k)
    # bar_labels = [gene_names[i] if gene_names and i < len(gene_names) else f"Gene{i}" for i in top_k_idx]

    # orig_vals = np.mean(Y_pred_orig.numpy(), axis=0)[top_k_idx]
    # ablt_vals = np.mean(Y_pred_ablate.numpy(), axis=0)[top_k_idx]

    # bar_width = 0.35
    # plt.bar(indices, orig_vals, width=bar_width, color='green', label='Original')
    # plt.bar(indices + bar_width, ablt_vals, width=bar_width, color='red', label='Ablated')

    # plt.xticks(indices + bar_width / 2, bar_labels, rotation=45)
    # plt.ylabel("Mean predicted expression")
    # plt.title(title_ or f"Expression change of top {top_k} genes\n(after ablating L={l_idx}, R={r_idx})")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Visualization: compare original vs ablated prediction
    up_idx = top_up_idx
    down_idx = top_down_idx

    orig = Y_pred_orig.numpy()
    ablt = Y_pred_ablate.numpy()

    up_orig = np.mean(orig[:, up_idx], axis=0)
    up_ablt = np.mean(ablt[:, up_idx], axis=0)

    down_orig = np.mean(orig[:, down_idx], axis=0)
    down_ablt = np.mean(ablt[:, down_idx], axis=0)

    up_labels = [gene_names[i] if gene_names and i < len(gene_names) else f"Gene{i}" for i in up_idx]
    down_labels = [gene_names[i] if gene_names and i < len(gene_names) else f"Gene{i}" for i in down_idx]

    fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)

    bar_width = 0.3
    x = np.arange(5)

    axes[0].bar(x, up_orig, width=bar_width, label='Original', color='green')
    axes[0].bar(x + bar_width, up_ablt, width=bar_width, label='Ablated', color='pink')
    axes[0].set_xticks(x + bar_width / 2)
    axes[0].set_xticklabels(up_labels, rotation=45)
    axes[0].set_title("Top 5 Upregulated Genes")
    axes[0].set_ylabel("Mean predicted expression")
    axes[0].legend(loc='upper right')

    axes[1].bar(x, down_orig, width=bar_width, label='Original', color='green')
    axes[1].bar(x + bar_width, down_ablt, width=bar_width, label='Ablated', color='skyblue')
    axes[1].set_xticks(x + bar_width / 2)
    axes[1].set_xticklabels(down_labels, rotation=45)
    axes[1].set_title("Top 5 Downregulated Genes")
    axes[1].legend(loc='upper right')

    plt.suptitle(title_ or f"Expression Change after Ablating L={l_idx}, R={r_idx}")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_file)
    plt.show()

    return {
        "top_k_idx": top_k_idx,
        "mean_delta_top_k": mean_delta[top_k_idx],
        "top_up_idx": top_up_idx,
        "top_down_idx": top_down_idx,
        "mean_delta": mean_delta
    }



def get_ablation_scores(model, T_receiver, l_idx, r_idx, gene_names=None):
    """
    Function to calculate the expression change scores (Δexpr) for every gene
    after ablation of the ligand-receptor interaction.
    
    Args:
    model: Trained model for gene expression prediction.
    T_receiver: Receiver cell tensor (input to the model).
    gene_names: List of gene names for visualization (optional).
    
    Returns:
    dict: A dictionary containing gene names and their corresponding Δexpr (change in expression).
    """
    model.eval()

    # Original input (before ablation)
    T_original = torch.tensor(T_receiver, dtype=torch.float32).view(T_receiver.shape[0], -1)

    # Ablated input (after ablation of the ligand-receptor pair)
    T_ablate = T_receiver.copy()
    T_ablate[:, l_idx, r_idx] = 0.0  # Knockout this communication pair
    T_ablate = torch.tensor(T_ablate, dtype=torch.float32).view(T_receiver.shape[0], -1)

    # Predict expression before and after ablation
    with torch.no_grad():
        Y_pred_orig = model(T_original)
        Y_pred_ablate = model(T_ablate)

    # Calculate expression change (Δexpr) for each gene
    delta = (Y_pred_orig - Y_pred_ablate).numpy()  # [num_cells, num_genes]
    mean_delta = np.mean(delta, axis=0)  # signed change (mean across cells)
    ads_delta = np.abs(mean_delta)

    # Create a dictionary of gene names (if provided) and their Δexpr values
    ablation_scores = {}
    for i, score in enumerate(ads_delta):
        gene_name = gene_names[i] if gene_names and i < len(gene_names) else f"Gene{i}"
        ablation_scores[gene_name] = score

    return ablation_scores
