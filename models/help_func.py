import numpy as np
import pandas as pd
import commot as ct
import scanpy as sc
from scipy import special
import scipy.sparse as sp
from mpl_chord_diagram import chord_diagram
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import networkx as nx
np.set_printoptions(precision=2)



# ================================================== For Methods ===========================================
# def diffusion_decrease_func(C, dist_thre, func_name='erfc', plot_=False):
#     Distance_D = pdist(C, metric='euclidean')
#     Distance_D = squareform(Distance_D)
#     # distance: scaled from 0 to 1
#     Distance_D = scale_data(Distance_D, 1, 0)

#     if func_name == 'erfc':
#         # distance: scaled from -3 to 3
#         rho = special.erfc(scale_data(Distance_D, 3, -3))
#         rho = scale_data(rho, 1, 0)
#         if plot_:
#             plt.scatter(Distance_D.flatten(), rho.flatten(), s=1, label='erfc')
#             plt.grid()
#             plt.ylabel('Decay weight')
#             plt.xlabel('Spot distance')
#             plt.legend()
#             plt.show()

#     elif func_name == 'sigmoid':
#         # distance: scaled from -6 to 6
#         rho = 1 - special.expit(scale_data(Distance_D, 6, -6))
#         rho = scale_data(rho, 1, 0)
#         if plot_:
#             plt.scatter(Distance_D.flatten(), rho.flatten(), s=1, label='sigmoid')
#             plt.grid()
#             plt.ylabel('Decay weight')
#             plt.xlabel('Spot distance')
#             plt.legend()
#             plt.show()

#     rho[Distance_D >= dist_thre] = 0

#     return sp.csc_matrix(rho)


def scale_data(x, max_val, min_val):
    """
    Linearly scale input array x to a new range [min_val, max_val].
    """
    x = np.array(x)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        return np.full_like(x, min_val)
    else:
        return (x - x_min) / (x_max - x_min) * (max_val - min_val) + min_val


def diffusion_decrease_func(C, m=5, platform=None, neighbor_k=5, plot_=False):
    """
    Compute spatial diffusion decay weights using erfc function.

    Parameters:
        C (ndarray): 2D coordinates of shape (n_samples, 2).
        m (int): Number of smallest non-zero distances to average for spot_size.
        platform (str): Platform name for default neighbor_k setting.
        neighbor_k (int): Multiplier for spot_size to define effective_range.
        plot_ (bool): Whether to visualize the decay curve.

    Returns:
        rho (csc_matrix): Sparse matrix of decay weights.
    """

    platform_defaults = {
        'Visium': 5,
        'Slide-seq': 30,
        'CosMx': 15,
        'Xenium': 20,
    }

    if platform in platform_defaults:
        neighbor_k = platform_defaults[platform]

    print(f'neighbor_k: {neighbor_k}')

    # 1. Compute pairwise Euclidean distance matrix
    Distance_D = squareform(pdist(C, metric='euclidean'))

    # 2. Extract m smallest non-zero distances and compute spot size
    triu_idx = np.triu_indices_from(Distance_D, k=1)
    all_nonzero_dists = Distance_D[triu_idx]
    min_m_dists = np.sort(all_nonzero_dists)[:m]
    spot_size = np.mean(min_m_dists)

    # 3. Define effective range
    effective_range = neighbor_k * spot_size

    # 4. Map distance [0, effective_range] to erfc input [0, 6] (i.e., [-3, 3])
    x_mapped = Distance_D * (6.0 / effective_range)
    x_mapped = np.clip(x_mapped, 0, 6)
    erfc_input = x_mapped - 3  # shift to [-3, 3]
    rho = special.erfc(erfc_input)

    # 5. Normalize to [0, 1]
    rho = scale_data(rho, max_val=1, min_val=0)

    # 6. Zero out beyond effective_range
    rho[Distance_D > effective_range] = 0

    # 7. Plot decay curve
    if plot_:
        D_flat = Distance_D.flatten()
        R_flat = rho.flatten()
        mask = D_flat <= effective_range
        x_vals = D_flat[mask]
        y_vals = R_flat[mask]

        plt.figure(figsize=(7, 4))
        plt.scatter(x_vals, y_vals, s=5, alpha=0.8, color='blue', label='Decay weights')

        # Mark every multiple of spot_size
        for i in range(neighbor_k + 1):
            dist = i * spot_size
            if dist <= effective_range:
                idx = np.argmin(np.abs(x_vals - dist))
                plt.scatter(x_vals[idx], y_vals[idx], color='red')
                plt.text(x_vals[idx], y_vals[idx] + 0.03, f'{i}×spot_size', ha='center', fontsize=8)

        plt.title('ERFC Decay Curve')
        plt.xlabel('Distance')
        plt.ylabel('Decay Weight')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return sp.csc_matrix(rho)


def sparse_array_repeat_by_mat(input_sp_mat, row_times, col_times):
    """
    Repeat sparse matrix by block concatenation.
    
    Parameters:
        input_sp_mat: sparse matrix
        row_times: int, number of vertical repetitions
        col_times: int, number of horizontal repetitions
    
    Returns:
        repeated sparse matrix
    """
    row_repeated = sp.vstack([input_sp_mat] * row_times)
    full_repeated = sp.hstack([row_repeated] * col_times)
    return full_repeated


def sparse_array_repeat_by_element(input_sp_mat, row_times, col_times):
    """
    Repeat each element of sparse matrix row_times x col_times times.
    
    Parameters:
        input_sp_mat: sparse matrix
        row_times: int
        col_times: int
    
    Returns:
        repeated sparse matrix
    """
    repeat_matrix_row = sp.csr_matrix(np.ones((row_times, 1)))
    repeat_matrix_col = sp.csr_matrix(np.ones((1, col_times)))
    repeated_matrix = sp.kron(sp.kron(input_sp_mat, repeat_matrix_row), repeat_matrix_col)
    return repeated_matrix


def create_mask_mat_M_T(rho, D, nl, ns, nr):
    """
    Create mask matrix M and weight matrix T from decay weights and LR database.
    
    Parameters:
        rho: sparse matrix of decay weights (ns x ns)
        D: DataFrame of ligand-receptor presence (nl x nr)
        nl, ns, nr: int, sizes of ligand, spots, receptor
    
    Returns:
        M: sparse mask matrix (nl*ns x nr*ns)
        T: sparse weight matrix (nl*ns x nr*ns)
    """
    D_sp = sp.csc_matrix(D.values)  # convert DataFrame to sparse matrix efficiently
    M = sparse_array_repeat_by_mat(D_sp, ns, ns).astype(int)
    T = sparse_array_repeat_by_element(rho, nl, nr).astype(np.float32)
    return sp.csc_matrix(M), sp.csc_matrix(T)


def create_A_B(nonzero_row_index, nonzero_col_index, nl, ns, nr):
    """
    Create sparse matrices A and B for the optimization problem based on nonzero indices.
    
    Parameters:
        nonzero_row_index: array-like, row indices of nonzero elements
        nonzero_col_index: array-like, column indices of nonzero elements
        nl, ns, nr: int
    
    Returns:
        A, B: sparse matrices in csc format
    """
    # Sort indices to ensure alignment
    help_row_index_df = pd.DataFrame(nonzero_row_index, columns=['idx']).sort_values('idx')
    help_col_index_df = pd.DataFrame(nonzero_col_index, columns=['idx']).sort_values('idx')

    data = np.ones(len(help_row_index_df), dtype=int)

    A = sp.csc_matrix((data, (help_row_index_df['idx'], help_row_index_df.index)),
                      shape=(nl * ns, len(help_row_index_df)))
    B = sp.csc_matrix((data, (help_col_index_df['idx'], help_col_index_df.index)),
                      shape=(nr * ns, len(help_col_index_df)))

    return A, B


def create_zeros_csc(n_row, n_col):
    x = sp.csc_array(([0.], ([0], [0])), shape=(n_row, n_col))
    return x - x


# def exp_for_csc(X_sp_csc):
#     row_index, col_index, x = sp.find(X_sp_csc)
#     X_sp_csc1 = sp.csc_array((np.exp(x), (row_index, col_index)), shape=X_sp_csc.shape)
#     return X_sp_csc1
# def exp_for_csc(X_sp_csc):
#     return sp.csc_matrix((np.exp(X_sp_csc.data), X_sp_csc.indices, X_sp_csc.indptr), shape=X_sp_csc.shape)
def exp_for_csc(X_sp_csc, clip_min=-50, clip_max=50):
    clipped_data = np.clip(X_sp_csc.data, clip_min, clip_max)
    return sp.csc_matrix((np.exp(clipped_data), X_sp_csc.indices, X_sp_csc.indptr), shape=X_sp_csc.shape)


# def log_for_csc(X_sp_csc):
#     row_index, col_index, x = sp.find(X_sp_csc)
#     X_sp_csc1 = sp.csc_array((np.log(x), (row_index, col_index)), shape=X_sp_csc.shape)
#     return X_sp_csc1
def log_for_csc(X_sp_csc):
    row_index, col_index, x = sp.find(X_sp_csc)
    mask = x > 0
    X_sp_csc1 = sp.csc_array((np.log(x[mask]), (row_index[mask], col_index[mask])), shape=X_sp_csc.shape)
    return X_sp_csc1



def H_entropy(x):
    x = x.toarray().flatten()
    return x.dot(np.log(x) - 1)


def create_LRDatabase_D(adata,
                        min_cell=None,
                        min_cell_pct=None,
                        database='CellChat',  # ‘CellChat’ or ‘CellPhoneDB_v4.0’.
                        species='human',
                        L_R_confirm=None,
                        kept_pathway=None,
                        save_file=None):
    df_CellChat = ct.pp.ligand_receptor_database(database=database,
                                                 species=species)
    if kept_pathway is not None:
        df_CellChat = df_CellChat[df_CellChat['2'].isin(kept_pathway)]

    if min_cell_pct is not None:
        df_CellChat_filtered = ct.pp.filter_lr_database(df_ligrec=df_CellChat,
                                                        adata=adata,
                                                        min_cell_pct=min_cell_pct)
    elif min_cell is not None:
        df_CellChat_filtered = ct.pp.filter_lr_database(df_ligrec=df_CellChat,
                                                        adata=adata,
                                                        min_cell=min_cell)
    else:
        df_CellChat_filtered = df_CellChat.copy()

    df_CellChat_filtered = df_CellChat_filtered.iloc[:, 0:-1]
    df_CellChat_filtered.columns = ['ligand', 'receptor', 'pathway']


    if L_R_confirm is not None:
        new_rows = []
        for temp_L_R_confirm in L_R_confirm:
            temp_l, temp_r = temp_L_R_confirm.split('_')
            new_rows.append({'ligand': temp_l, 'receptor': temp_r, 'pathway': temp_L_R_confirm})
        
        df_CellChat_filtered = pd.concat([df_CellChat_filtered, pd.DataFrame(new_rows)], ignore_index=True)

    df_CellChat_filtered_new = pd.DataFrame(columns=['ligand', 'receptor', 'pathway'])
    for i in range(df_CellChat_filtered.shape[0]):
        temp_ligand = df_CellChat_filtered['ligand'].iloc[i].split('_')
        temp_receptors = df_CellChat_filtered['receptor'].iloc[i].split('_')
        temp_pathway = df_CellChat_filtered['pathway'].iloc[i]

        new_rows = []
        for lig in temp_ligand:
            for recep in temp_receptors:
                new_rows.append({'ligand': lig, 'receptor': recep, 'pathway': temp_pathway})

        df_CellChat_filtered_new = pd.concat([df_CellChat_filtered_new, pd.DataFrame(new_rows)], ignore_index=True)

    df_CellChat_filtered_new = df_CellChat_filtered_new.drop_duplicates()

    if save_file is not None:
        df_CellChat_filtered_new.to_csv(save_file)

    df_CellChat_filtered_new['exist'] = 1

    LRDatabase_D = pd.pivot_table(df_CellChat_filtered_new,
                                  index='ligand', columns='receptor', values='exist',
                                  aggfunc='mean')
    LRDatabase_D = LRDatabase_D.fillna(0)

    return LRDatabase_D


def LiANA_LRDatabase_D(adata,
                       min_cell=None,
                       min_cell_pct=None,
                       species='human',
                       save_file=None):
    import liana as li
    if species == 'human':
        resource = li.rs.select_resource('consensus')
    elif species == 'mouse':
        resource = li.rs.select_resource('mouseconsensus')

    if min_cell_pct is not None:
       resource_filtered = ct.pp.filter_lr_database(df_ligrec=resource, adata=adata, min_cell_pct=min_cell_pct)

    if min_cell is not None:
       resource_filtered = ct.pp.filter_lr_database(df_ligrec=resource, adata=adata, min_cell=min_cell)

    else:
        resource_filtered = resource.copy()
    resource_filtered.columns = ['ligand', 'receptor']

    if save_file is not None:
        resource_filtered.to_csv(save_file)

    resource_filtered['exist'] = 1
    
    LRDatabase_D = pd.pivot_table(resource_filtered,
                                  index='ligand', columns='receptor', values='exist',
                                  aggfunc='mean')
    LRDatabase_D = LRDatabase_D.fillna(0)

    return LRDatabase_D 













# =================================================== For Plots ==============================================
def custom_cmap(min_col='blue',
                cen_col='white',
                max_col='red',
                center=0.5):
    custom_cmap_ = mcolors.LinearSegmentedColormap.from_list(
        name="custom_cmap",
        colors=[(0, min_col), (center, cen_col), (1, max_col)]
    )
    return custom_cmap_

def custom_palette():
    # Custom palette based on the provided hex color values
    custom_palette_hex = [
        "#E6AACE", "#B14D89",
        "#F0E39D", "#AE9E41",
        "#CFF27E", "#85AB2E",
        "#AAADE2", "#6367BA",
        "#CB797A", "#A23032",
        "#D5C3B5", "#A57751"
    ]
    return custom_palette_hex


def chord_plot(data_np, data_name, my_camp=None,
               gap=0.03, chord_width=0.8, directed=True):
    if my_camp is None:
        my_camp = mcolors.ListedColormap(['#B5B4B4', '#13B388', '#33ABD3', '#CB4E3E'])

    chord_diagram(
        data_np, data_name, gap=gap, chordwidth=chord_width,
        use_gradient=True, directed=directed,
        cmap=my_camp,
        rotate_names=False, fontcolor='grey',
    )


def sankey_plot(CCC_data1,
                title_text, save_file,
                font_size=15,
                node_colors=None):
    if node_colors is None:
        node_colors = ['#E78F8E', '#5DB7DE', '#FFD700', '#4DAA57', '#9370DB', 'orange'] * 2
    CCC_data = CCC_data1.copy()
    # Define the labels for each node
    label = list(CCC_data.index) * 2
    node_colors = node_colors

    CCC_data.index = range(CCC_data.shape[0])
    CCC_data.columns = range(CCC_data.shape[0])
    data_df = unpivot(CCC_data)

    # Define the source nodes
    source = data_df['index']
    link_colors = [node_colors[src] for src in source]

    # Define the target nodes
    target = data_df['col'] + CCC_data.shape[0]

    # Define the values for the flows
    value = data_df['value']

    # Create a Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            x=[0.25] * 6 + [0.75] * 6,
            # y=np.tile(np.linspace(0.1, 0.9, 6), 2),
            color=node_colors,
            label=label
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )))

    fig.update_layout(title_text=title_text, font_size=font_size)
    fig.write_image(save_file)


def matrix_plot(adata_, vmax=None, vmin=None, figsize=(6, 4), rotation_=45):
    sc.tl.rank_genes_groups(adata_, 'Cell_type_interaction', method='t-test')
    DE_result = adata_.uns['rank_genes_groups']
    DE_genes = pd.DataFrame(DE_result['names'])
    # sc.pl.dotplot(adata_for_diff, np.array(DE_genes.iloc[0, :]).T.flatten(),
    #               groupby='Cell_type_interaction')

    adata_.layers['scaled'] = sc.pp.scale(adata_, copy=True).X  # 对列计算z-score

    fig, axs = plt.subplots(figsize=figsize)

    if vmax is None:
        ax_dict = sc.pl.matrixplot(
            adata_, np.array(DE_genes.iloc[0, :]),
            ax=axs,
            groupby='Cell_type_interaction', dendrogram=False,
            colorbar_title='mean z-score', layer='scaled',
            cmap='RdBu_r', show=False)
        for ax in ax_dict.values():
            for label in ax.get_xticklabels():
                label.set_rotation(rotation_)

        plt.tight_layout()
        plt.show()

    else:
        ax_dict = sc.pl.matrixplot(
            adata_, np.array(DE_genes.iloc[0, :]),
            ax=axs,
            groupby='Cell_type_interaction', dendrogram=False,
            colorbar_title='mean z-score', layer='scaled',
            vmax=vmax, vmin=vmin, cmap='RdBu_r', show=False)
        for ax in ax_dict.values():
            for label in ax.get_xticklabels():
                label.set_rotation(rotation_)

        plt.tight_layout()
        plt.show()


def plot_graph_with_categories(G, color_map):
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, node_size=5, node_color=color_map, with_labels=False)
    plt.show()


def plot_violin_boxes(adata_df, genes, groupby, violin_colors, box_darken_amount=0.2, 
                      figsize=(12, 3), savefile='./violinplot.svg'):
    import seaborn as sns
    from matplotlib.colors import to_rgba, rgb_to_hsv, hsv_to_rgb
    from scipy.stats import mannwhitneyu

    def darken_color(color, amount=0.3):
        color = to_rgba(color)
        hsv = rgb_to_hsv(color[:3])
        hsv[2] *= (1 - amount)
        return hsv_to_rgb(hsv)

    fig, axes = plt.subplots(1, len(genes), figsize=figsize)

    if len(genes) == 1: 
        axes = [axes]

    for ax, gene in zip(axes, genes):
        box_colors = [darken_color(color, box_darken_amount) for color in violin_colors]

        sns.violinplot(x=groupby, y=gene, data=adata_df, ax=ax, inner=None, palette=violin_colors)
        sns.boxplot(x=groupby, y=gene, data=adata_df, ax=ax, fliersize=0, width=0.2, palette=box_colors)

        for collection in ax.collections:
            collection.set_edgecolor("none")

        # Calculate p-values and add annotations
        unique_groups = adata_df[groupby].unique()
        text_positions = [adata_df[gene].max() * (1 + 0.5 * i) for i in range(len(unique_groups))]
        for i, group1 in enumerate(unique_groups[:-1]):
            for j, group2 in enumerate(unique_groups[i+1:], start=i+1):
                group1_data = adata_df[adata_df[groupby] == group1][gene]
                group2_data = adata_df[adata_df[groupby] == group2][gene]
                _, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                # Display the p-value in scientific notation
                mid_point = (i + j) / 2
                ax.text(mid_point, text_positions[min(i, j)], f'p={p_value:.1e}', horizontalalignment='center', color='black')

        ax.set_xlabel(' ')
        ax.set_ylabel(gene)

    plt.tight_layout()
    plt.savefig(savefile)
    plt.show()











# =============================================== For benchamrk ===============================================
def normalize_matrix(matrix, tol=1e-8):
    from sklearn.preprocessing import minmax_scale
    """Min-max normalize a matrix to [0, 1] range"""
    flat = matrix.flatten().astype(float)
    if np.ptp(flat) > tol:
        scaled = minmax_scale(flat)
    else:
        scaled = flat
    return scaled.reshape(matrix.shape)

def normalize_methods(method_results, interaction_list):
    for method, result in method_results.items():
        filtered_result = {
            lr_pair: normalize_matrix(result[lr_pair])
            for lr_pair in interaction_list
            if lr_pair in result
        }
        method_results[method] = filtered_result
    return method_results


# def align_and_stack_methods(method_results, interaction_list):
#     from collections import defaultdict
#     import numpy as np
#     """
#     Align methods on the same L-R keys and normalize each matrix.
    
#     Returns:
#     - aligned: dict of L_R -> list of aligned and normalized matrices
#     """
#     aligned = defaultdict(list)
#     for lr_pair in interaction_list:
#         for method, result in method_results.items():
#             if lr_pair in result:
#                 mat = normalize_matrix(result[lr_pair])
#             else:
#                 mat = np.zeros_like(next(iter(result.values())))
#             aligned[lr_pair].append(mat)
#     return aligned

def align_and_stack_methods(method_results, interaction_list):
    from collections import defaultdict
    import numpy as np
    aligned = defaultdict(list)
    for lr_pair in interaction_list:
        for method, result in method_results.items():
            aligned[lr_pair].append(result[lr_pair])
    return aligned


# def is_cell_level(matrix, shape):
#     return matrix.shape[0] == shape 

# def compute_cell_level_consensus(aligned_dict, method_names, ncells):
#     """
#     Filter aligned results to retain only those from cell-level methods.
    
#     Parameters:
#     - aligned_dict: dict of L_R -> list of matrices
#     - method_names: list of method names aligned to each matrix
    
#     Returns:
#     - consensus: dict of L_R -> list of cell-level matrices only
#     """
#     consensus = {}
#     for lr, matrices in aligned_dict.items():
#         cell_matrices = [
#             mat for mat, name in zip(matrices, method_names)
#             if is_cell_level(mat, ncells)
#         ]
#         if len(cell_matrices) > 0:
#             consensus[lr] = cell_matrices

#         consensus[lr] = np.mean(cell_matrices, axis=0)
#     return consensus


# def compute_group_level_consensus(method_results, cell_to_group, ncells):
#     """
#     Compute group-level consensus matrix under each L-R pair.
#     Supports both cell-level and group-level matrices from different methods.
    
#     Parameters:
#     - method_results: dict of method_name -> {L_R -> matrix}
#     - cell_to_group: dict of cell_name -> group_name

#     Returns:
#     - group_consensus: dict of L_R -> group-group matrix (numpy array)
#     """
#     from collections import defaultdict

#     group_matrices = defaultdict(list)
    
#     # Get all L-R keys across methods
#     all_lr_keys = set()
#     for result in method_results.values():
#         all_lr_keys.update(result.keys())

#     for lr in all_lr_keys:
#         for method, result in method_results.items():
#             if lr in result:
#                 mat = result[lr]
#                 # If matrix shape matches cell count → cell-level
#                 if mat.shape[0] == ncells and mat.shape[1] == ncells:
#                     mat_group = aggregate_to_group(mat, cell_to_group).values
#                 else:
#                     # Already group-level, use as is
#                     mat_group = mat if isinstance(mat, np.ndarray) else mat.values
#                 mat_group = normalize_matrix(mat_group)
#                 group_matrices[lr].append(mat_group)
        
#     group_consensus = {
#         lr: np.mean(mats, axis=0) for lr, mats in group_matrices.items()
#     }

#     return group_consensus


# def compute_group_level_consensus(aligned_dict, method_names, cell_to_group, ncells):
#     """
#     Compute group-level consensus matrix under each L-R pair from aligned_dict.

#     Parameters:
#     - aligned_dict: dict of L_R -> list of matrices (some may be cell-level, some group-level)
#     - method_names: list of method names aligned to each matrix
#     - cell_to_group: dict of cell_name -> group_name
#     - ncells: int, number of cells (to distinguish resolution)

#     Returns:
#     - consensus: dict of L_R -> group-level consensus matrix (numpy array)
#     """
#     consensus = {}

#     for lr, matrices in aligned_dict.items():
#         group_matrices = []
#         for mat, name in zip(matrices, method_names):
#             if is_cell_level(mat, ncells):
#                 mat_group = aggregate_to_group(mat, cell_to_group)
#                 mat_group = mat_group.values  # convert to ndarray
#             else:
#                 mat_group = mat if isinstance(mat, np.ndarray) else mat.values

#             mat_group = normalize_matrix(mat_group)
#             group_matrices.append(mat_group)

#         if group_matrices:
#             consensus[lr] = np.mean(group_matrices, axis=0)

#     return consensus


def evaluate_method_vs_consensus(
    method_results,
    consensus_dict,
    lr_list=None,
    metric="pearson",
    k=100
):
    """
    Evaluate similarity between each method and the consensus using a chosen metric.

    Parameters:
    - method_results: dict of method_name -> {L_R -> matrix (DataFrame or ndarray)}
    - consensus_dict: dict of L_R -> consensus matrix (ndarray or DataFrame)
    - lr_list: optional list of L_R to restrict evaluation
    - metric: one of ["pearson", "spearman", "topk"]
    - k: for "topk", number of top elements to consider

    Returns:
    - pd.DataFrame: rows = method, columns = L_R pairs, values = similarity scores
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    scores = {}
    lr_keys = lr_list if lr_list is not None else consensus_dict.keys()

    for method, result in method_results.items():
        method_scores = {}

        for lr in lr_keys:
            if lr not in result or lr not in consensus_dict:
                continue

            mat = result[lr]
            consensus = consensus_dict[lr]

            # Convert DataFrame to array if needed
            mat = mat.values if isinstance(mat, pd.DataFrame) else mat
            consensus = consensus.values if isinstance(consensus, pd.DataFrame) else consensus

            try:
                if metric == "pearson":
                    score, _ = pearsonr(mat.flatten(), consensus.flatten())

                elif metric == "spearman":
                    score, _ = spearmanr(mat.flatten(), consensus.flatten())

                elif metric == "topk":
                    flat_method = mat.flatten()
                    flat_consensus = consensus.flatten()

                    topk_method = np.argsort(flat_method)[-k:]
                    topk_consensus = np.argsort(flat_consensus)[-k:]
                    score = len(set(topk_method) & set(topk_consensus)) / k

                else:
                    raise ValueError(f"Unsupported metric: {metric}")

                method_scores[lr] = score

            except Exception:
                method_scores[lr] = np.nan

        scores[method] = method_scores

    return pd.DataFrame(scores).T  # methods as rows


def evaluate_method_with_expression(
    adata_lr,
    adata_expr,
    methods=["pearson", "spearman"],
):
    """
    Evaluate similarity between L_R send/receive scores and corresponding gene expression.

    Parameters:
    - adata_lr: AnnData, with var names like "L_R_receive" or "L_R_send"
    - adata_expr: AnnData, gene expression matrix
    - methods: list of correlation methods to use. Options: "pearson", "spearman", "kendall"

    Returns:
    - pd.DataFrame with columns: L, R, type, method, correlation, pvalue
    """
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr, kendalltau

    method_map = {
        "pearson": pearsonr,
        "spearman": spearmanr,
        "kendall": kendalltau,
    }

    results = []

    # Match cells
    assert all(adata_lr.obs_names == adata_expr.obs_names), "Cell names must match"

    for var in adata_lr.var_names:
        ligand, receptor, mode = var.split("_")

        gene = ligand if mode == "send" else receptor

        x = adata_lr[:, var].X.flatten()
        y = adata_expr[:, gene].X.A.flatten()

        for method in methods:
            corr_func = method_map[method]
            corr, pval = corr_func(x, y)
            results.append({
                "L": ligand,
                "R": receptor,
                "type": mode,
                "method": method,
                "correlation": corr,
                "pvalue": pval
            })

    return pd.DataFrame(results)


def evaluate_spatial_similarity_with_expression(
    adata_lr,
    adata_expr,
    methods=["ssim", "emd"],
    normalize=True
):
    import pandas as pd
    from skimage.metrics import structural_similarity as ssim
    from scipy.stats import wasserstein_distance
    from scipy.spatial import distance_matrix

    coords = adata_lr.obsm["spatial"]

    results = []

    for var in adata_lr.var_names:
        ligand, receptor, mode = var.split("_")
        gene = ligand if mode == "send" else receptor

        v1 = adata_lr[:, var].X.flatten()
        v2 = adata_expr[:, gene].X.A.flatten()

        if normalize:
            v1 = (v1 - v1.min()) / (v1.max() - v1.min() + 1e-8)
            v2 = (v2 - v2.min()) / (v2.max() - v2.min() + 1e-8)

        # Project variables onto a spatial grid (generate a diagram at the minimum resolution)
        try:
            from scipy.sparse import coo_matrix

            df = pd.DataFrame({
                'x': coords[:, 0],
                'y': coords[:, 1],
                'v1': v1,
                'v2': v2
            })

            # Map to integer coordinate grid
            df['x'] = df['x'] - df['x'].min()
            df['y'] = df['y'] - df['y'].min()
            df['x'] = df['x'].astype(int)
            df['y'] = df['y'].astype(int)

            grid_shape = (df['y'].max() + 1, df['x'].max() + 1)
            grid1 = coo_matrix((df['v1'], (df['y'], df['x'])), shape=grid_shape).toarray()
            grid2 = coo_matrix((df['v2'], (df['y'], df['x'])), shape=grid_shape).toarray()

        except Exception as e:
            continue

        for method in methods:
            score = None
            try:
                if method == "ssim":
                    score = ssim(grid1, grid2, data_range=1.0)

                elif method == "emd":
                    score = wasserstein_distance(v1, v2)

                else:
                    continue

                results.append({
                    "L": ligand,
                    "R": receptor,
                    "type": mode,
                    "method": method,
                    "similarity": score
                })
            except Exception as e:
                results.append({
                    "L": ligand,
                    "R": receptor,
                    "type": mode,
                    "method": method,
                    "similarity": None,
                    "error": str(e)
                })

    return pd.DataFrame(results)


def compute_spatial_autocorrelation(adata, n_neighbors=10):
    from libpysal.weights import KNN
    from esda.moran import Moran
    import pandas as pd
    import numpy as np
    """
    Compute Moran's I spatial autocorrelation for each L-R feature.

    Parameters:
    - adata: AnnData with obsm['spatial'] and X (cells x L-R)
    - n_neighbors: number of neighbors for spatial graph

    Returns:
    - DataFrame with L-R as index and Moran's I and p-values as columns
    """
    coords = adata.obsm['spatial']
    results = {}

    # Build spatial weight graph using k-nearest neighbors
    w = KNN.from_array(coords, k=n_neighbors)
    w.transform = 'r'

    for i, lr in enumerate(adata.var_names):
        x = adata.X[:, i].flatten()
        if np.all(x == x[0]):  # constant vector
            moran_i, p_val = np.nan, np.nan
        else:
            mi = Moran(x, w)
            moran_i, p_val = mi.I, mi.p_norm

        results[lr] = {'moran_I': moran_i, 'p_value': p_val}

    return pd.DataFrame.from_dict(results, orient='index')


def rank_genes_by_total_signal(adata):
    import pandas as pd
    import numpy as np
    """
    Rank genes (L-R features) by total signal intensity across all cells.

    Parameters:
    - adata: AnnData with X (cells x L-R)

    Returns:
    - DataFrame with L-R as index and total signal as column, sorted by total signal
    """
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    total_signal = X.sum(axis=0)

    results = pd.DataFrame({
        'total_signal': total_signal
    }, index=adata.var_names)

    return results.sort_values(by='total_signal', ascending=False)


def compute_group_distance_matrix(cell_coords, cell_to_group):
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist
    """
    Construct a group-group distance matrix by averaging cell-cell distances.

    Parameters:
    - cell_coords: (N, 2) numpy array of spatial coordinates
    - cell_to_group: dict mapping cell name -> group label

    Returns:
    - group_dist_df: group × group DataFrame of mean distances
    """

    # Get consistent cell names and order
    cells = list(cell_to_group.keys())
    coords = cell_coords[[cells.index(c) for c in cells]]
    groups = [cell_to_group[c] for c in cells]
    
    df = pd.DataFrame(coords, index=cells, columns=['x', 'y'])
    df['group'] = groups

    # Compute cell-cell distance matrix
    cell_dist = pd.DataFrame(cdist(df[['x', 'y']], df[['x', 'y']]), index=cells, columns=cells)

    # Add group info to dist matrix
    cell_dist['source_group'] = df['group']
    grouped_rows = cell_dist.groupby('source_group')

    group_dist = {}
    for g1, g1_block in grouped_rows:
        temp = g1_block.drop(columns='source_group')
        temp.columns.name = None
        temp = temp.T
        temp['target_group'] = df['group']
        temp_grouped = temp.groupby('target_group').mean().T
        group_dist[g1] = temp_grouped.mean()

    group_dist_df = pd.DataFrame(group_dist)
    group_dist_df = group_dist_df.T  # make it symmetric
    group_dist_df = group_dist_df.loc[sorted(group_dist_df.index), sorted(group_dist_df.columns)]

    # Zero out diagonal (optional)
    np.fill_diagonal(group_dist_df.values, 0)

    return group_dist_df


def evaluate_spatial_CCCdiff_nearfar(lr_matrices, cell_coords, distance_quantile=0.5):
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist
    from scipy.stats import mannwhitneyu
    """
    Evaluate spatial dependence of communication matrices across L-R pairs.

    Parameters:
    - lr_matrices: dict of {lr_name: cell-cell matrix (2D np.ndarray or pd.DataFrame)}
    - cell_coords: np.ndarray or pd.DataFrame of shape (n_cells, 2)
    - distance_quantile: float, threshold to divide near vs far (e.g., 0.5 for median)

    Returns:
    - pd.DataFrame with rows = LR names and columns = ['p_value', 'near_mean', 'far_mean', 'distance_threshold']
    """
    results = []

    n = cell_coords.shape[0]
    distance_matrix = cdist(cell_coords, cell_coords, metric='euclidean')
    mask = ~np.eye(n, dtype=bool)
    distance_flat = distance_matrix[mask]
    distance_thresh = np.quantile(distance_flat, distance_quantile)

    for lr_name, mat in lr_matrices.items():
        comm_flat = mat[mask]
        near_vals = comm_flat[distance_flat <= distance_thresh]
        far_vals = comm_flat[distance_flat > distance_thresh]


        # p_val = mannwhitneyu(near_vals, far_vals, alternative='two-sided').pvalue
        near_mean = np.mean(near_vals)
        far_mean = np.mean(far_vals)

        results.append({
            'LR': lr_name,
            # 'p_value': p_val,
            'near_mean': near_mean,
            'far_mean': far_mean,
            'diff': near_mean-far_mean,
            'distance_threshold': distance_thresh
        })

    return pd.DataFrame(results).set_index('LR')


def evaluate_spatial_CCCdiff_nearfar_CT(lr_matrices, group_dist_df, distance_quantile=0.5):
    results = []

    n = group_dist_df.shape[0]
    mask = ~np.eye(n, dtype=bool)
    distance_flat = group_dist_df.values[mask]
    distance_thresh = np.quantile(distance_flat, distance_quantile)

    for lr_name, mat in lr_matrices.items():
        comm_flat = mat[mask]
        near_vals = comm_flat[distance_flat <= distance_thresh]
        far_vals = comm_flat[distance_flat > distance_thresh]

        near_mean = np.mean(near_vals)
        far_mean = np.mean(far_vals)

        results.append({
            'LR': lr_name,
            'near_mean': near_mean,
            'far_mean': far_mean,
            'distance_threshold': distance_thresh
        })

    return pd.DataFrame(results).set_index('LR')


# --- Core Utilities ---
def compute_spatial_weight_matrix(coords):
    from scipy.spatial.distance import cdist
    """Compute spatial weight matrix: inverse of distance"""
    dists = cdist(coords, coords)
    return 1 / (dists + 1e-4)


def compute_co_localization_matrix(lig_expr, rec_expr, coords):
    """Compute co-localization interaction: outer product x spatial weight"""
    weights = compute_spatial_weight_matrix(coords)
    interaction = np.outer(lig_expr, rec_expr)
    return interaction * weights


def compute_spatial_cosine(pred_matrix, truth_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    """Compute cosine similarity between two matrices"""
    pred_flat = pred_matrix.flatten().reshape(1, -1)
    truth_flat = truth_matrix.flatten().reshape(1, -1)
    return cosine_similarity(pred_flat, truth_flat)[0][0]


# --- Core evaluation of a single pair ---
def evaluate_interaction(pred_matrix, truth_matrix, thre_perc=10):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import minmax_scale
    """
    Compute multiple metrics between predicted interaction matrix and spatial co-localization matrix
    """
    pred = pred_matrix.flatten()
    truth = truth_matrix.flatten()

    pred_norm = minmax_scale(pred)
    truth_norm = minmax_scale(truth)

    # Top thre_perc% of true co-localization is labeled as 1
    threshold = np.percentile(truth_norm, 100-thre_perc)
    labels = (truth_norm >= threshold).astype(int)

    auc = roc_auc_score(labels, pred_norm)
    auprc = average_precision_score(labels, pred_norm)
    cosine_sim = compute_spatial_cosine(pred_matrix, truth_matrix)

    return {
        'AUROC': auc,
        'AUPR': auprc,
        'Cosine Similarity': cosine_sim,
    }

def build_expr_dict_from_adata(adata, lr_pairs, sep='_'):
    """
    Build expression dict at cell resolution
    """
    expr_dict = {}
    genes = set(adata.var_names)

    for lr in lr_pairs:
        ligand, receptor = lr.split(sep)
        if ligand in genes and receptor in genes:
            lig_expr = adata[:, ligand].X
            rec_expr = adata[:, receptor].X

            # Convert sparse to dense
            if hasattr(lig_expr, 'toarray'):
                lig_expr = lig_expr.toarray().flatten()
            else:
                lig_expr = np.asarray(lig_expr).flatten()

            if hasattr(rec_expr, 'toarray'):
                rec_expr = rec_expr.toarray().flatten()
            else:
                rec_expr = np.asarray(rec_expr).flatten()

            expr_dict[lr] = (lig_expr, rec_expr)
    return expr_dict



def build_group_expr_and_coords(adata, group_key, lr_pairs, sep='_', coord_strategy='medoid'):
    from scipy.spatial.distance import cdist
    """
    Build group-level expression and group spatial coordinates (mean or medoid)

    Parameters:
    - adata: AnnData with spatial info in obsm['spatial']
    - group_key: column in adata.obs indicating group label
    - lr_pairs: list of ligand_receptor strings
    - sep: separator between ligand and receptor in string
    - coord_strategy: 'mean' or 'medoid'

    Returns:
    - group_expr: dict {lr_pair: (lig_expr_vector, rec_expr_vector)}
    - coords_group: np.array of shape [n_groups, 2]
    """
    group_expr = {}
    group_coords = []
    groups = sorted(adata.obs[group_key].unique().tolist())

    for g in groups:
        idx = adata.obs[group_key] == g
        coords = adata.obsm['spatial'][idx]

        if coord_strategy == 'mean':
            group_coord = coords.mean(axis=0)

        elif coord_strategy == 'medoid':
            dist_mat = cdist(coords, coords)
            total_dists = dist_mat.sum(axis=1)
            group_coord = coords[np.argmin(total_dists)]

        group_coords.append(group_coord)

    coords_group = np.vstack(group_coords)

    for lr in lr_pairs:
        ligand, receptor = lr.split(sep)
        ligs, recs = [], []

        for g in groups:
            idx = adata.obs[group_key] == g
            sub = adata[idx]

            lig_data = sub[:, ligand].X
            rec_data = sub[:, receptor].X

            if hasattr(lig_data, 'toarray'):
                lig_val = lig_data.toarray().mean()
                rec_val = rec_data.toarray().mean()
            else:
                lig_val = np.asarray(lig_data).mean()
                rec_val = np.asarray(rec_data).mean()

            ligs.append(lig_val)
            recs.append(rec_val)

        if len(ligs) == len(groups):
            group_expr[lr] = (np.array(ligs), np.array(recs))

    return group_expr, coords_group



def evaluate_method_set(method_results, expr_dict, coords, thre_perc=10):
    """
    Evaluate all methods in a single resolution setting (cell-level or group-level)
    Parameters:
        method_results: {method_name: {L_R: pred_matrix}}
        expr_dict: {L_R: (lig_expr, rec_expr)}
        coords: numpy array [n, 2] matching expression vectors
    Returns:
        DataFrame with all evaluation metrics
    """
    results = []

    for method, lr_dict in method_results.items():
        for lr_pair, pred_matrix in lr_dict.items():

            lig_expr, rec_expr = expr_dict[lr_pair]
            truth_matrix = compute_co_localization_matrix(lig_expr, rec_expr, coords)

            scores = evaluate_interaction(pred_matrix, truth_matrix, thre_perc)
            scores.update({'method': method, 'lr_pair': lr_pair})
            results.append(scores)

    return pd.DataFrame(results)


def compute_consensus_agreement(method_results, threshold_percent=10, vote_ratio=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score
    """
    Compute AUROC and AUPR between each method and a consensus binary interaction matrix.

    Parameters:
    - method_results: dict of method -> {LR_pair: CCC matrix}
    - threshold_percent: float, top k% values in a matrix are considered high-interaction (1)
    - vote_ratio: float in (0,1], proportion of methods that must agree to call an interaction "consensus"

    Returns:
    - DataFrame: rows = methods, columns = AUROC, AUPR
    """
    from collections import defaultdict

    lr_pairs = list(next(iter(method_results.values())).keys())
    methods = list(method_results.keys())
    results = defaultdict(list)
    consensus_binary_dict = {}

    for lr in lr_pairs:
        # Step 1: collect all matrices for this L-R
        matrices = []
        for method in methods:
            mat = method_results[method].get(lr)
            if mat is None:
                break
            matrices.append(np.array(mat))
        
        if len(matrices) != len(methods):
            continue  # Skip this LR if any method missing

        # Step 2: binarize each matrix at top k%
        binary_matrices = []
        for mat in matrices:
            flat = mat.flatten()
            thresh = np.percentile(flat, 100 - threshold_percent)
            binary = (mat >= thresh).astype(int)
            binary_matrices.append(binary)

        # Step 3: majority voting to get consensus binary matrix
        binary_stack = np.stack(binary_matrices)  # shape: (n_methods, N, N)
        vote_threshold = int(np.ceil(len(methods) * vote_ratio))
        consensus_binary = (binary_stack.sum(axis=0) >= vote_threshold).astype(int)
        consensus_flat = consensus_binary.flatten()
        consensus_binary_dict[lr] = consensus_binary

        # Step 4: evaluate each method vs consensus
        for method, mat in zip(methods, matrices):
            pred_flat = mat.flatten()
            try:
                auc = roc_auc_score(consensus_flat, pred_flat)
            except ValueError:
                auc = np.nan
            try:
                aupr = average_precision_score(consensus_flat, pred_flat)
            except ValueError:
                aupr = np.nan

            results['method'].append(method)
            results['lr_pair'].append(lr)
            results['AUROC'].append(auc)
            results['AUPR'].append(aupr)

    # Step 5: average across LR pairs
    df = pd.DataFrame(results)
    return df, consensus_binary_dict













# ===============================================Format change ===========================================
def aggregate_to_group(matrix, cell_to_group):
    """
    Convert a cell-cell matrix to group-group matrix by averaging.

    Returns:
    - group-level matrix as a pandas DataFrame
    """
    all_groups = sorted(set(cell_to_group.values()))

    df = pd.DataFrame(matrix, index=cell_to_group.keys(), columns=cell_to_group.keys())  # cell*cell
    df['source_group'] = df.index.map(cell_to_group)

    numeric_cols = df.columns.drop('source_group')

    grouped = df.groupby('source_group')[numeric_cols].apply(
        lambda x: x.T.groupby(cell_to_group).mean().T
    )
    res = grouped.groupby(level=0).mean()
    res = res[all_groups].loc[all_groups]

    return res


def build_cell_lr_tensor(cell_level_result, ncells):
    """
    Construct a cell * L * R tensor from cell-cell interaction matrices.

    Parameters:
    - cell_level_result: dict of 'L_R' -> cell * cell matrix
    - ncells: number of cells

    Returns:
    - tensor: numpy array of shape (ncells, n_ligands, n_receptors)
    - ligands: list of ligand names
    - receptors: list of receptor names
    """
    import numpy as np

    lr_keys = sorted(cell_level_result.keys())
    
    ligands = []
    receptors = []
    cell_by_lr = []

    for lr in lr_keys:
        mat = cell_level_result[lr]  # shape: ncells x ncells
        # Column sums represent each cell's score as a receiver
        vec = mat.sum(axis=0)  # shape: (ncells,)
        cell_by_lr.append(vec)

        ligand, receptor = lr.split("_", 1)
        ligands.append(ligand)
        receptors.append(receptor)

    # Stack into shape: (n_lr, ncells)
    cell_by_lr = np.stack(cell_by_lr)  # shape: (n_lr, ncells)

    # Transpose to shape: (ncells, n_lr)
    cell_by_lr = cell_by_lr.T

    # Construct 3D tensor
    unique_ligands = sorted(set(ligands))
    unique_receptors = sorted(set(receptors))

    ligand_idx = {lig: i for i, lig in enumerate(unique_ligands)}
    receptor_idx = {rec: i for i, rec in enumerate(unique_receptors)}

    tensor = np.zeros((ncells, len(unique_ligands), len(unique_receptors)))

    for lr_i, (lig, rec) in enumerate(zip(ligands, receptors)):
        i = ligand_idx[lig]
        j = receptor_idx[rec]
        tensor[:, i, j] = cell_by_lr[:, lr_i]

    return tensor, unique_ligands, unique_receptors


def tensor_to_method_result(tensor, ligands, receptors):
    """
    Convert a 4D CCC tensor to method_results format.
    
    Parameters:
    - tensor: numpy array of shape [n_cell, n_cell, n_ligand, n_receptor]
    - ligands: list of ligand names, len == n_ligand
    - receptors: list of receptor names, len == n_receptor
    
    Returns:
    - method_result: dict of L_R -> cell × cell matrix
    """
    n_ligand = len(ligands)
    n_receptor = len(receptors)
    
    result = {}
    for i in range(n_ligand):
        for j in range(n_receptor):
            lr_key = f"{ligands[i]}_{receptors[j]}"
            result[lr_key] = tensor[:, :, i, j]
    return result


def filter_method_result_by_LR_database(method_result, LR_database):
    """
    Filter method_result by known valid L-R pairs from LR_database.

    Parameters:
    - method_result: dict of 'Ligand_Receptor' -> matrix
    - LR_database: pandas DataFrame, rows=ligands, cols=receptors, 1 means valid

    Returns:
    - filtered_method_result: dict with only valid L-R pairs
    """
    valid_pairs = set()

    for ligand in LR_database.index:
        for receptor in LR_database.columns:
            if LR_database.loc[ligand, receptor] == 1:
                key = f"{ligand}_{receptor}"
                valid_pairs.add(key)

    filtered_result = {
        lr: mat for lr, mat in method_result.items() if lr in valid_pairs
    }

    return filtered_result

def tensor_to_ccc_dataframe(
    tensor: np.ndarray,
    LR_mask: pd.DataFrame,
    cell_names: None
) -> pd.DataFrame:
    """
    Convert a 4D cell-cell-ligand-receptor tensor to a DataFrame.
    Only ligand-receptor pairs with value==1 in LR_mask are included.
    Row and column labels use 'source->target' string format.

    Parameters:
    - tensor: np.ndarray with shape (n_cell_i, n_cell_j, n_ligand, n_receptor)
    - LR_mask: pd.DataFrame with ligands as index and receptors as columns; values of 1 are retained
    - cell_names: optional list of cell names; if None, default names like 'C0', 'C1'... are used

    Returns:
    - pd.DataFrame with rows like 'cellA->cellB' and columns like 'ligand->receptor'
    """

    n_cell_i, n_cell_j, n_ligand, n_receptor = tensor.shape

    assert LR_mask.shape == (n_ligand, n_receptor), "LR_mask must match tensor ligand-receptor dimensions"

    # Generate default cell names if not provided
    if cell_names is None:
        cell_names = [f"C{i}" for i in range(n_cell_i)]

    ligand_names = LR_mask.index.to_list()
    receptor_names = LR_mask.columns.to_list()

    # Get valid ligand-receptor pairs from LR_mask
    allowed_pairs = np.argwhere(LR_mask.values == 1)

    # Row index: format 'cellA->cellB'
    row_labels = [
        f"{cell_names[i]}->{cell_names[j]}"
        for i in range(n_cell_i)
        for j in range(n_cell_j)
    ]

    # Columns: format 'ligand->receptor'
    col_labels = []
    data = []

    for k, m in allowed_pairs:
        values = tensor[:, :, k, m].reshape(-1)
        data.append(values)
        col_labels.append(f"{ligand_names[k]}->{receptor_names[m]}")

    # Create DataFrame
    df = pd.DataFrame(np.column_stack(data), index=row_labels, columns=col_labels)

    return df

def long_df_to_group_matrices(df, all_groups=None):
    """
    Convert long-format group-level CCC DataFrame (with L-R) to method_result format.

    Parameters:
    - df: DataFrame with columns ['ligand', 'receptor', 'source', 'target', 'score']
    - all_groups: optional list of all group names for consistent matrix shape

    Returns:
    - method_result: dict of 'Ligand_Receptor' -> group × group matrix (DataFrame)
    """
    method_result = {}

    grouped = df.groupby(['ligand', 'receptor'])

    for (ligand, receptor), sub_df in grouped:
        lr_key = f"{ligand}_{receptor}"
        

        pivot = (
            sub_df
            .groupby(['source', 'target'])['score']
            .mean()
            .unstack(fill_value=0)
        )

        if all_groups is not None:
            pivot = pivot.reindex(index=all_groups, columns=all_groups, fill_value=0)

        method_result[lr_key] = pivot.values

    return method_result


def build_adata_from_ccc_receive(ccc_dict):
        
    import pandas as pd
    import numpy as np
    import anndata as ad
    """
    Construct AnnData object from CCC dict by computing reception intensity (column sums).

    Parameters:
    - ccc_dict: dict of LR_pair -> CCC matrix (n_cells x n_cells)

    Returns:
    - AnnData object with shape (n_cells, n_LR_pairs)
    """
    reception_vectors = {}

    for lr, mat in ccc_dict.items():
        mat = mat.values if isinstance(mat, pd.DataFrame) else mat
        
        # Column sum = total signal received per cell
        reception = mat.sum(axis=0)  # shape: (n_cells,)
        reception_vectors[lr] = reception

    # Build (n_cells x n_LR) matrix
    reception_df = pd.DataFrame(reception_vectors)

    # Create AnnData
    adata = ad.AnnData(X=reception_df.values)
    adata.var_names = reception_df.columns
    adata.obs_names = reception_df.index.astype(str)

    return adata


def build_adata_from_ccc_send_receive(ccc_dict):
    import pandas as pd
    import anndata as ad

    """
    Construct AnnData object from CCC dict by computing both sending and receiving intensity.

    Parameters:
    - ccc_dict: dict of LR_pair -> CCC matrix (n_cells x n_cells)

    Returns:
    - AnnData object with shape (n_cells, n_LR_pairs * 2)
      Columns named like "LR1_receive", "LR1_send", etc.
    """
    reception_vectors = {}
    sending_vectors = {}

    for lr, mat in ccc_dict.items():
        mat = mat.values if isinstance(mat, pd.DataFrame) else mat

        reception = mat.sum(axis=0)  # shape: (n_cells,)
        reception_vectors[f"{lr}_receive"] = reception

        sending = mat.sum(axis=1)   # shape: (n_cells,)
        sending_vectors[f"{lr}_send"] = sending

    combined_df = pd.DataFrame({**reception_vectors, **sending_vectors})
    
    # Create AnnData
    adata = ad.AnnData(X=combined_df.values)
    adata.var_names = combined_df.columns
    adata.obs_names = combined_df.index.astype(str)

    return adata


def dict_to_ccc_dataframe_CT(lr_dict: dict, cell_classes, cell_names=None):
    import numpy as np
    import pandas as pd
    from itertools import product
    """
    Constructs a dataframe summarizing ligand-receptor communication 
    strengths between cell type pairs.

    Parameters:
    ----------
    lr_dict : dict
        Dictionary where key is 'Ligand-Receptor' string, value is [n_cells x n_cells] numpy array.
    cell_classes : list or pd.Series
        List or Series of length n_cells representing cell type for each cell.
    cell_names : list (optional)
        List of cell names of length n_cells. If None, uses default index [0, 1, 2, ...].

    Returns:
    -------
    pd.DataFrame
        A dataframe where rows are (class_i, class_j) and columns are LR pairs,
        containing the average communication strength for each LR pair.
    """
    n_cells = len(cell_classes)
    cell_classes = pd.Series(cell_classes)

    # Use default names if cell_names not provided
    if cell_names is None:
        cell_names = list(range(n_cells))
    else:
        if len(cell_names) != n_cells:
            raise ValueError("Length of cell_names must match number of cells.")
    
    # Create mapping from cell name to class
    cell_class_map = dict(zip(cell_names, cell_classes))

    # Generate all possible cell-cell pairs using custom names
    cell_pairs = list(product(cell_names, repeat=2))
    df_base = pd.DataFrame(cell_pairs, columns=["cell_i", "cell_j"])

    # Map class labels
    df_base["class_i"] = df_base["cell_i"].map(cell_class_map)
    df_base["class_j"] = df_base["cell_j"].map(cell_class_map)

    # Add LR communication values
    for lr, matrix in lr_dict.items():
        if matrix.shape != (n_cells, n_cells):
            raise ValueError(f"Matrix for {lr} has shape {matrix.shape}, expected ({n_cells}, {n_cells})")
        df_base[lr] = matrix.flatten()

    # Drop raw cell names — not needed after mapping
    df_base = df_base.drop(columns=["cell_i", "cell_j"])

    # Group by class pair and average
    grouped = df_base.groupby(["class_i", "class_j"]).mean()

    return grouped



def compute_intra_inter_mean(grouped_df):
    import pandas as pd
    """
    Computes the mean communication strength of each LR pair separately
    for intra-class and inter-class pairs, without applying any threshold.

    Parameters:
    ----------
    grouped_df : pd.DataFrame
        DataFrame from build_lr_dataframe, with MultiIndex (class_i, class_j)
        and LR pairs as columns.

    Returns:
    -------
    dict
        {
            'intra': pd.Series,  # mean values of LR pairs for class_i == class_j
            'inter': pd.Series   # mean values of LR pairs for class_i != class_j
        }
    """
    # Reset index to access class_i and class_j for filtering
    df = grouped_df.reset_index()

    # Determine which rows are intra-class vs inter-class
    intra_df = df[df["class_i"] == df["class_j"]]
    inter_df = df[df["class_i"] != df["class_j"]]

    # Select only the LR columns
    lr_cols = grouped_df.columns  # All columns except the index are LR pairs

    # Compute mean across rows
    intra_means = intra_df[lr_cols].mean()
    inter_means = inter_df[lr_cols].mean()

    return pd.DataFrame({
        "intra": intra_means,
        "inter": inter_means
    })










# ================================================== utils ===============================================
def unpivot(frame):
    N, K = frame.shape
    data_unpivot = {
        "value": frame.to_numpy().ravel("F"),
        "col": np.asarray(frame.columns).repeat(N),
        "index": np.tile(np.asarray(frame.index), K),
    }
    return pd.DataFrame(data_unpivot, columns=["index", "col", "value"])

def coordinate_rotation(angle, vertices):
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_vertices = np.dot(vertices, rotation_matrix)
    return rotated_vertices


def calculate_sequence_entropy(sequence):
    sequence = sequence[sequence>1e-10]
    import numpy as np
    from scipy.stats import entropy
    value, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / len(sequence)
    seq_entropy = entropy(probabilities, base=2)  
    return seq_entropy


def set_seed(seed: int = 42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_dominant_strength(adata, send_suffix="_send", receive_suffix="_receive", output_layer="dominant"):
    """
    Compute dominant communication strength for each LR pair based on columns
    in adata.X (expression matrix), where columns are named with *_send and *_receive.

    Stores result in `adata.obsm[output_layer]` as a DataFrame with one column per LR.

    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object with LR data in adata.X and names in adata.var_names
    send_suffix : str
        Suffix for sending columns (default: '_send')
    receive_suffix : str
        Suffix for receiving columns (default: '_receive')
    output_layer : str
        Name to store results in adata.obsm[output_layer]
    """
    import pandas as pd
    import numpy as np
    from scipy.sparse import issparse

    adata_new = adata.copy()

    var_names = adata_new.var_names.tolist()

    send_cols = [name for name in var_names if name.endswith(send_suffix)]
    receive_cols = [name for name in var_names if name.endswith(receive_suffix)]

    lr_names = set(name.replace(send_suffix, "") for name in send_cols) & \
               set(name.replace(receive_suffix, "") for name in receive_cols)


    # Prepare results
    result_df = pd.DataFrame(index=adata_new.obs_names)

    for lr in lr_names:
        send_name = lr + send_suffix
        recv_name = lr + receive_suffix

        send_idx = var_names.index(send_name)
        recv_idx = var_names.index(recv_name)

        send_vals = adata_new.X[:, send_idx].toarray().flatten() if issparse(adata_new.X) else adata_new.X[:, send_idx]
        recv_vals = adata_new.X[:, recv_idx].toarray().flatten() if issparse(adata_new.X) else adata_new.X[:, recv_idx]

        # dominant: send if higher, else -receive
        dominant = np.where(send_vals >= recv_vals, send_vals, -recv_vals)

        result_df[f'{lr}_dominant'] = dominant

    adata_new.obs[result_df.columns] = result_df
    
    return adata_new