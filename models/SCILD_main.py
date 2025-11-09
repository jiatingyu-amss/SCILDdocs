from models.help_func import *
import matplotlib.pylab as plt
np.set_printoptions(precision=2)




# ---------------------------------------------------------------------------------
#                                 The SCILD method
# --------------------------------------------------------------------------------
class SCILD:
    def __init__(
        self,
        adata: any = None,
        LRDatabase_D: any = None,
        neighbor_k: int = 5,
        platform: any = None,
        alpha_q: float = 0.1,
        alpha_f: float = 0.1,
        alpha_g: float = 0.1,
        niter_max: int = 100,
        eps: float = 1e-5,
        verbose: bool = False,
        plot_error: bool = False,
        plot_save_path: str = None
    ):

        """
        :param adata: Anndata type, contains expression matrices and spatial spot information of spatial transcriptomics data.
        :param LRDatabase_D: A ligand-receptor interaction database, represented as a data frame with elements being 0 or 1, sized ligand*receptor.
        :param dist_threshold (float): A normalized distance threshold parameter. Interactions are not considered if the spot distance exceeds this threshold, which ranges from 0 to 1.
        :param alpha_q (float): Coefficient for the non-negative penalty term in the optimization objective function, default is 0.1.
        :param alpha_f (float): Coefficient for the relaxation term of the ligand emission signal constraint in the optimization objective function, default is 0.1.
        :param alpha_g (float): Coefficient for the relaxation term of the receptor reception signal constraint in the optimization objective function, default is 0.1.
        :param niter_max (int): Maximum number of iterations for solving the optimization problem, default is 100.
        :param eps (float): Iteration termination error for solving the optimization problem, default is 1e-5.
        :param verbose (bool): Flag to print the process details during execution.
        :param plot_error (bool): Flag to plot the error during the iterative solving process.
        """

        # check
        assert adata.X.format == 'csc' or adata.X.format == 'csr', \
            'please make sure that the format of adata.X is csc or csr!'

        # import data
        self.adata = adata

        self.LRDatabase_D = LRDatabase_D  # size: ligand*receptor
        # self.Expression_E = pd.DataFrame(self.adata.X.todense(),      # size: spot*gene
        #                                  index=self.adata.obs.index,
        #                                  columns=self.adata.var.index)
        self.Expression_E = self.adata.to_df()
        self.Coordinate_C = pd.DataFrame(self.adata.obsm['spatial'],  # size: spot*2
                                         index=self.adata.obs.index,
                                         columns=['x', 'y'])

        # import parameter
        self.neighbor_k = neighbor_k
        self.platform = platform
        self.alpha_q = alpha_q
        self.alpha_f = alpha_f
        self.alpha_g = alpha_g
        self.niter_max = niter_max
        self.eps = eps
        self.verbose = verbose
        self.plot_error = plot_error
        self.plot_save_path = plot_save_path

        # make sure that expression data are available for ligands and receptors
        common_G_R = sorted(list(set(self.LRDatabase_D.columns).intersection(self.Expression_E.columns)))
        common_G_L = sorted(list(set(self.LRDatabase_D.index).intersection(self.Expression_E.columns)))
        self.LRDatabase_D = self.LRDatabase_D[common_G_R]
        self.LRDatabase_D = self.LRDatabase_D.loc[common_G_L]

        # delete useless ligands and receptors
        self.LRDatabase_D = self.LRDatabase_D.loc[(self.LRDatabase_D != 0).any(axis=1)]
        self.LRDatabase_D = self.LRDatabase_D.loc[:, (self.LRDatabase_D != 0).any(axis=0)]
        self.LRDatabase_D.sort_index(axis=0)
        self.LRDatabase_D.sort_index(axis=1)

        # record the actual number of spots, genes, ligands, receptors
        self.nl, self.nr = self.LRDatabase_D.shape
        self.ns, self.ng = self.Expression_E.shape
        assert self.Coordinate_C.shape == (self.ns, 2)

        # define variables
        self.rho = None
        self.M = None
        self.T = None
        self.t = None
        self.A = None
        self.B = None
        self.nonzero_row_index = None
        self.nonzero_col_index = None

        self.E1 = None
        self.E2 = None

        self.P = None
        self.Q = None
        self.xP = None
        self.xQ = None
        self.f = None
        self.g = None



    def preparing(self):
        print("*************Preparing*************")
        # define rho(csc), M(csc)
        self.rho = diffusion_decrease_func(C=np.array(self.Coordinate_C), neighbor_k=self.neighbor_k, platform=self.platform)
        self.M, self.T = create_mask_mat_M_T(self.rho, self.LRDatabase_D, self.nl, self.ns, self.nr)

        # define E1(array), E2(array)
        self.E1 = np.array(self.Expression_E[self.LRDatabase_D.index]).reshape(-1, 1)
        self.E2 = np.array(self.Expression_E[self.LRDatabase_D.columns]).reshape(-1, 1)

        # # record row_index, col_index of nonzero element (by col)
        # self.nonzero_row_index, self.nonzero_col_index, _ = sp.find(self.M)

        # ----------------------
        # # define t(csc), A(csc), B(csc)
        # # self.t = sp.csc_matrix(self.T.multiply(self.M)[self.nonzero_row_index, self.nonzero_col_index].reshape(-1, 1))
        # vals = self.T[self.nonzero_row_index, self.nonzero_col_index].A1  
        # self.t = sp.csc_matrix(vals.reshape(-1, 1))

        M_csc = self.M.tocsc()  

        nonzero_rows = []
        nonzero_cols = []

        for col in range(M_csc.shape[1]):
            start_ptr = M_csc.indptr[col]
            end_ptr = M_csc.indptr[col + 1]
            rows = M_csc.indices[start_ptr:end_ptr]
            nonzero_rows.append(rows)
            nonzero_cols.append(np.full_like(rows, col))

        self.nonzero_row_index = np.concatenate(nonzero_rows)
        self.nonzero_col_index = np.concatenate(nonzero_cols)

        vals = self.T[self.nonzero_row_index, self.nonzero_col_index].A1
        self.t = sp.csc_matrix(vals.reshape(-1, 1))
        # ----------------------

        self.A, self.B = create_A_B(self.nonzero_row_index, self.nonzero_col_index,
                                    self.nl, self.ns, self.nr)

        self.B = self.B.multiply(self.t.T)  # Hadamard product for sparse matrix

        self.A = sp.csc_matrix(self.A)
        self.B = sp.csc_matrix(self.B)



    def solving_optimization(self, mu0, v0):
        self.mu0=mu0
        self.v0=v0
        print("\n*************Solving*************")
        self.mu, self.v = solve_multipliers(
            t=self.t,
            A=self.A,
            B=self.B,
            E1=self.E1,
            E2=self.E2,
            nl=self.nl,
            nr=self.nr,
            ns=self.ns,
            alpha_q=self.alpha_q,
            alpha_f=self.alpha_f,
            alpha_g=self.alpha_g,
            niter_max=self.niter_max,
            eps=self.eps,
            mu0=self.mu0,
            v0=self.v0,
            verbose=self.verbose,
            plot_error=self.plot_error,
            plot_save_path=self.plot_save_path
        )

        # compute Xp, Xq
        self.xQ = np.array(np.exp((self.t - self.A.T.dot(self.mu) - self.B.T.dot(self.v)) / self.alpha_q))
        self.xP = self.t.toarray() * self.xQ
        self.f = self.alpha_f / (self.mu + 1e-18)
        self.g = self.alpha_g / (self.v + 1e-18)

        # recover P, Q
        self.P = sp.csc_array((self.xP.flatten(), (self.nonzero_row_index, self.nonzero_col_index)),
                              shape=self.M.shape)
        self.Q = sp.csc_array((self.xQ.flatten(), (self.nonzero_row_index, self.nonzero_col_index)),
                              shape=self.M.shape)

        self.adata.obsm['sum-sender-Q'] = pd.DataFrame(index=self.adata.obs.index)
        self.adata.obsm['sum-sender-P'] = pd.DataFrame(index=self.adata.obs.index)
        self.adata.obsm['sum-receiver'] = pd.DataFrame(index=self.adata.obs.index)


    def query_LR(self, query_LR_name):
        L_name = query_LR_name[0]
        R_name = query_LR_name[1]
        L_index = list(self.LRDatabase_D.index).index(L_name)
        R_index = list(self.LRDatabase_D.columns).index(R_name)

        row_index = [L_index + i * self.nl for i in range(self.ns)]
        col_index = [R_index + i * self.nr for i in range(self.ns)]

        query_mat_P = self.P[row_index, :]
        query_mat_P = query_mat_P[:, col_index]

        query_mat_Q = self.Q[row_index, :]
        query_mat_Q = query_mat_Q[:, col_index]

        self.adata.obsm['sum-sender-P'][L_name + '_' + R_name] = query_mat_P.sum(1)
        self.adata.obsm['sum-sender-Q'][L_name + '_' + R_name] = query_mat_Q.sum(1)
        self.adata.obsm['sum-receiver'][L_name + '_' + R_name] = query_mat_P.sum(0)

        return query_mat_P.astype('float32'), L_name + '_' + R_name
    

    def query_all_LR(self):
        self.tensor_P = self.P.toarray().reshape(
            self.ns, self.nl, self.ns, self.nr).transpose(0, 2, 1, 3)
        self.tensor_Q = self.Q.toarray().reshape(
            self.ns, self.nl, self.ns, self.nr).transpose(0, 2, 1, 3)
        

    def select_strong_LRs(self, agg='sum'):
        """
        Calculate and rank ligand-receptor pairs based on overall communication strength,
        considering only LR pairs with LRDatabase_D == 1.

        Args:
            agg (str): Aggregation method: 'sum' (default) or 'mean'

        Returns:
            pd.DataFrame: DataFrame with columns ['Ligand', 'Receptor', 'Score'], sorted descending by Score
        """
        import pandas as pd

        results = []
        for l in range(self.nl):
            for r in range(self.nr):
                # Only include L-R pairs marked as valid in LRDatabase_D
                if self.LRDatabase_D.iloc[l, r] != 1:
                    continue

                mat = self.tensor_P[:, :, l, r]  # ns x ns matrix

                # Aggregate
                if agg == 'sum':
                    score = mat.sum()
                elif agg == 'mean':
                    score = mat.mean()
                else:
                    raise ValueError("agg must be 'sum' or 'mean'")

                results.append({
                    "Ligand": self.LRDatabase_D.index[l],
                    "Receptor": self.LRDatabase_D.columns[r],
                    "Score": score,
                })

        # Sort and return
        df = pd.DataFrame(results)
        df = df.sort_values('Score', ascending=False).reset_index(drop=True)

        return df


    def store_spatial_LR_to_obs(self, lr_df, key_prefix="LR"):
        """
        Store receiver-side communication vectors into `self.adata.obs` as pseudo-gene expressions
        for given ligand-receptor pairs.

        Args:
            lr_df (pd.DataFrame): DataFrame containing columns ['Ligand', 'Receptor'] specifying LR pairs to store.
            key_prefix (str): Prefix for keys stored in obs.
        """
        for _, row in lr_df.iterrows():
            l_name = row["Ligand"]
            r_name = row["Receptor"]
            l_idx = list(self.LRDatabase_D.index).index(l_name)
            r_idx = list(self.LRDatabase_D.columns).index(r_name)

            mat = self.tensor_P[:, :, l_idx, r_idx]
            pseudo_expr = mat.sum(axis=0).astype(np.float32)

            obs_key = f"{key_prefix}_{l_name}_{r_name}"
            self.adata.obs[obs_key] = pseudo_expr


    def merge_into_group(self, query_mat, group_name, remove_diag=True, group_func='mean'):
        query_mat_temp = query_mat.copy()
        query_mat_temp[group_name] = self.adata.obs[group_name]

        if group_func == 'mean':
            query_mat_temp = query_mat_temp.groupby(group_name).mean().T
            query_mat_temp[group_name] = self.adata.obs[group_name]
            query_mat_temp = query_mat_temp.groupby(group_name).mean().T

        elif group_func == 'sum':
            query_mat_temp = query_mat_temp.groupby(group_name).sum().T
            query_mat_temp[group_name] = self.adata.obs[group_name]
            query_mat_temp = query_mat_temp.groupby(group_name).sum().T

        if remove_diag is True:
            query_mat_temp = query_mat_temp - np.diag(np.diag(query_mat_temp))

        return query_mat_temp
    
    
    def compute_TSSR(self):
        """
        For each allowed ligand-receptor pair, compute:
        - total signal sent by each cell (sum over receivers)
        - total signal received by each cell (sum over senders)
        
        Return a DataFrame with rows = cells, columns = [L->R (send), L->R (recv)]
        """

        ligand_names = self.LRDatabase_D.index.to_list()
        receptor_names = self.LRDatabase_D.columns.to_list()
        cell_names = self.adata.obs.index

        # Find allowed ligand-receptor pairs
        allowed_pairs = np.argwhere(self.LRDatabase_D.values == 1)

        # Initialize result dict
        result = {cell: {} for cell in cell_names}

        for k, m in allowed_pairs:
            lig = ligand_names[k]
            rec = receptor_names[m]
            label_send = f"{lig}->{rec} (S)"
            label_recv = f"{lig}->{rec} (R)"

            # tensor[:, :, k, m] is [cell_i, cell_j]
            mat = self.tensor_P[:, :, k, m]

            # Sum along rows for senders, columns for receivers
            send_vector = mat.sum(axis=1)  # shape: [cell_i]
            recv_vector = mat.sum(axis=0)  # shape: [cell_j]

            for idx, cell in enumerate(cell_names):
                result[cell][label_send] = send_vector[idx]
                result[cell][label_recv] = recv_vector[idx]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(result, orient="index")
        df.index.name = "cell"

        return df




def solve_multipliers(t, A, B, E1, E2, nl, nr, ns,
                      alpha_q=0.1,
                      alpha_f=0.1,
                      alpha_g=0.1,
                      niter_max=100,
                      eps=1e-5,
                      mu0=None,
                      v0=None,
                      verbose=True,
                      plot_error=False,
                      plot_save_path=None):

    if mu0 is None:
        mu = np.random.random(nl * ns).reshape(-1, 1)
        # mu = np.zeros(nl * ns).reshape(-1, 1)
    else:
        mu = mu0.copy()

    if v0 is None:
        v = np.random.random(nr * ns).reshape(-1, 1)
        # v = np.zeros(nr * ns).reshape(-1, 1)
    else:
        v = v0.copy()



    # ==================================== iteration based method ========================================
    error_relative = [1]
    niter = 0

    while niter < niter_max and error_relative[-1] > eps:
        mu_previous = mu.copy()
        v_previous = v.copy()

        temp_xq = sp.csc_matrix(exp_for_csc(
            (t - A.T * (sp.csr_matrix(mu_previous)) - B.T * (sp.csr_matrix(v_previous))) / alpha_q
        ))

        # mu = mu_previous + alpha_f * (
        #         np.log(np.exp(-mu_previous / alpha_f) + (A * temp_xq).toarray()) - np.log(E1 + 1e-18)
        # )
        # v = v_previous + alpha_g * (
        #         np.log(np.exp(-v_previous / alpha_g) + (B * temp_xq).toarray()) - np.log(E2 + 1e-18)
        # )

        mu_input = -mu_previous / alpha_f
        mu_rhs = (A @ temp_xq).toarray() + 1e-18
        mu = mu_previous + alpha_f * (
            np.logaddexp(mu_input, np.log(mu_rhs)) - np.log(E1 + 1e-18)
        )

        v_input = -v_previous / alpha_g
        v_rhs = (B @ temp_xq).toarray() + 1e-18
        v = v_previous + alpha_g * (
            np.logaddexp(v_input, np.log(v_rhs)) - np.log(E2 + 1e-18)
        )

        error1 = abs(mu_previous - mu).max() / max(abs(mu_previous).max(), abs(mu).max(), 1e-18)
        error2 = abs(v_previous - v).max() / max(abs(v_previous).max(), abs(v).max(), 1e-18)
        error_relative.append((error1 + error2) / 2)

        niter = niter + 1

        if verbose:
            if niter % 10 == 0:
                print('The relative error is: ' + str(error_relative[-1]))


    if verbose:
        print('\n The final relative error is: ' + str(error_relative[-1]))
        print('The total iteration step is: ' + str(niter))

    if plot_error:
        plt.figure(figsize=(3, 3))
        plt.plot(error_relative[1:])
        plt.xlabel('Iteration step')
        plt.ylabel('Relative error')

        if plot_save_path is not None:
            plt.savefig(plot_save_path, bbox_inches='tight')

    return mu, v
