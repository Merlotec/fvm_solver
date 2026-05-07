import torch
from abc import ABC, abstractmethod

from time_fvm.utils.ell_kernel import csr_to_ell, ell_spmm, ell_spmv


class SPM(ABC):
    """ Sparse matrix. """
    def __init__(self, A, device):
        self.shape = A.shape
        self.device = device

    @abstractmethod
    def spMV(self, x: torch.Tensor, b: torch.Tensor=None) -> torch.Tensor:
        """ Sparse matrix-vector multiply: Ax+b """
        raise NotImplementedError("SPM is an abstract base class")

    @abstractmethod
    def spMM(self, X: torch.Tensor, b: torch.Tensor=None) -> torch.Tensor:
        """ Sparse matrix-matrix multiply: Ax+b"""
        raise NotImplementedError("SPM is an abstract base class")


class SPMCuda(SPM):
    """ Use triton accelerated ELL format for sparse matrix operations on CUDA. """
    def __init__(self, A: torch.Tensor, device):
        super().__init__(A, device)
        if A.layout != torch.sparse_csr:
            A = A.to_sparse_csr()

        A = A.to(device)
        self.vals, self.cols = csr_to_ell(A)

    def spMM(self, X: torch.Tensor, b: torch.Tensor=None) -> torch.Tensor:
        return ell_spmm(self.vals, self.cols, X, b=b)

    def spMV(self, x: torch.Tensor, b: torch.Tensor=None) -> torch.Tensor:
        return ell_spmv(self.vals, self.cols, x, b=b)


class SPMGeneral(SPM):
    """ Normal sparse matrix, directly using pytorch sparse operations. """
    def __init__(self, A: torch.Tensor, device):
        super().__init__(A, device)
        if A.layout != torch.sparse_csr:
            A = A.to_sparse_csr()

        A = A.to(device)
        self.A = torch.sparse_csr_tensor(A.crow_indices().to(torch.int32), A.col_indices().to(torch.int32), A.values(),
                                         size=A.size(), device=device)

    def spMM(self, X: torch.Tensor, b: torch.Tensor=None) -> torch.Tensor:
        if b is None:
            return torch.sparse.mm(self.A, X)
        else:
            return torch.addmm(b, self.A, X)

    def spMV(self, x: torch.Tensor, b: torch.Tensor=None) -> torch.Tensor:
        if b is None:
            return torch.mv(self.A, x)
        else:
            return torch.addmv(b, self.A, x)


def to_sparse(A: torch.Tensor, device) -> SPM:
    device = torch.device(device)
    if device.type == "cuda":
        return SPMCuda(A, device)
    else:
        return SPMGeneral(A, device)


def to_csr(A: torch.Tensor, device):
    """ Convert a dense matrix to sparse CSR format """
    if A.layout != torch.sparse_csr:
        A = A.to_sparse_csr()

    A = A.to(device)
    return torch.sparse_csr_tensor(A.crow_indices().to(torch.int32), A.col_indices().to(torch.int32), A.values(), size=A.size(), device=device)


def combine_facet_operators(A_main, A_bc, b_bc, bc_edge_mask, n_edges, n_cells, n_comp, device):
    """
    Combines a main-edge operator and a boundary-edge operator into a single global operator.

    Parameters:
      A_main      : sparse COO tensor of shape (n_edges_m*n_comp, n_cells*n_comp)
                    -- the main-edge operator (with local row ordering).
      A_bc        : sparse COO tensor of shape (n_edges_bc*n_comp, n_cells*n_comp)
                    -- the boundary-edge operator (with local row ordering).
      b_bc        : tensor of shape (n_edges_bc*n_comp,)
                    -- the offset vector for boundary edges.
      bc_edge_mask: Boolean tensor of shape (n_edges,)
                    -- True if the global edge is a boundary edge.
      n_edges     : int, total number of global edges.
      n_cells     : int, number of cells.
      n_comp      : int, number of components.
      device      : torch.device

    Returns:
      A_all       : sparse COO tensor of shape (n_edges*n_comp, n_cells*n_comp)
                    -- the combined operator.
      b_all       : tensor of shape (n_edges*n_comp,)
                    -- the combined offset vector.
    """

    """# Get the COO indices and values for the two operators.
    # (They must be in COO format.)
    A_main_indices = A_main._indices()  # shape (2, L_main)
    A_main_values = A_main._values()  # shape (L_main,)
    A_bc_indices = A_bc._indices()  # shape (2, L_bc)
    A_bc_values = A_bc._values()  # shape (L_bc,)

    # We will build lists of row indices, column indices, and values for the global operator.
    global_rows = []
    global_cols = []
    global_vals = []
    b_all_list = []  # offset for each global row

    # Counters for the local row index in A_main and A_bc.
    # They indicate which main (or bc) edge (block) we are currently processing.
    main_counter = 0
    bc_counter = 0

    # Loop over all global edges.
    for i in range(n_edges):
        # For each edge, process all components.
        for c in range(n_comp):
            # Compute the flattened (global) row index for edge i and component c.
            global_row = i * n_comp + c

            if not bc_edge_mask[i]:
                # --- Main edge ---
                # The corresponding local row in A_main is:
                local_row = main_counter * n_comp + c
                # Find the entries in A_main corresponding to this local row.
                mask = (A_main_indices[0, :] == local_row)
                # (These entries come with column indices and values.)
                cols = A_main_indices[1, :][mask]
                vals = A_main_values[mask]
                # Append these entries, but with the global row instead of the local row.
                for col, val in zip(cols.tolist(), vals.tolist()):
                    global_rows.append(global_row)
                    global_cols.append(col)
                    global_vals.append(val)
                # For a main edge, no offset is added.
                b_all_list.append(0.0)
            else:
                # --- Boundary edge ---
                local_row = bc_counter * n_comp + c
                mask = (A_bc_indices[0, :] == local_row)
                cols = A_bc_indices[1, :][mask]
                vals = A_bc_values[mask]
                for col, val in zip(cols.tolist(), vals.tolist()):
                    global_rows.append(global_row)
                    global_cols.append(col)
                    global_vals.append(val)
                # The offset for boundary edges comes from b_bc.
                # (Assume b_bc is a 1D tensor of length n_edges_bc*n_comp.)
                b_all_list.append(b_bc[local_row].item())
        # Update the local counters.
        if not bc_edge_mask[i]:
            main_counter += 1
        else:
            bc_counter += 1


    # Convert the lists into tensors.
    indices = torch.tensor([global_rows, global_cols], dtype=torch.long, device=device)
    values = torch.tensor(global_vals, dtype=A_main_values.dtype, device=device)

    # The global operator acts on flattened cell fields of length n_cells*n_comp and produces
    # an output of length n_edges*n_comp.
    size_all = (n_edges * n_comp, n_cells * n_comp)
    A_all = torch.sparse_coo_tensor(indices, values, size=size_all).coalesce()
    b_all = torch.tensor(b_all_list, dtype=A_main_values.dtype, device=device)"""

    # Assume the following inputs are given:
    # A_main: sparse COO tensor of shape (n_edges_m*n_comp, n_cells*n_comp)
    # A_bc: sparse COO tensor of shape (n_edges_bc*n_comp, n_cells*n_comp)
    # b_bc: tensor of shape (n_edges_bc*n_comp,)
    # bc_edge_mask: Boolean tensor of shape (n_edges,), where True indicates a boundary edge.
    # n_edges, n_cells, n_comp, device are given.

    # First, compute the mapping for global main and boundary edges.
    # The ordering of the local operators corresponds to the order of the global edges.
    main_edge_global_indices = torch.where(~bc_edge_mask)[0]  # shape: (n_main_edges,)
    bc_edge_global_indices = torch.where(bc_edge_mask)[0]  # shape: (n_bc_edges,)

    A_main_indices = A_main._indices()  # shape: (2, L_main)
    A_main_values = A_main._values()  # shape: (L_main,)
    A_bc_indices = A_bc._indices()  # shape: (2, L_bc)
    A_bc_values = A_bc._values()  # shape: (L_bc,)

    # --- Process A_main ---
    # For each local row in A_main, determine its main edge index and component:
    local_rows_main = A_main_indices[0, :]  # local row indices in A_main (range: 0 to n_edges_m*n_comp - 1)
    j_main = local_rows_main // n_comp  # index into main_edge_global_indices
    c_main = local_rows_main % n_comp  # component index

    # Map to global row index: for main edges the global row is (global_edge_index * n_comp + component)
    global_rows_main = main_edge_global_indices[j_main] * n_comp + c_main
    global_cols_main = A_main_indices[1, :]


    # --- Process A_bc ---
    local_rows_bc = A_bc_indices[0, :]  # local row indices in A_bc (range: 0 to n_edges_bc*n_comp - 1)
    j_bc = local_rows_bc // n_comp  # index into bc_edge_global_indices
    c_bc = local_rows_bc % n_comp  # component index

    global_rows_bc = bc_edge_global_indices[j_bc] * n_comp + c_bc
    global_cols_bc = A_bc_indices[1, :]

    # --- Combine the main and boundary contributions ---
    global_rows = torch.cat([global_rows_main, global_rows_bc], dim=0)
    global_cols = torch.cat([global_cols_main, global_cols_bc], dim=0)
    global_vals = torch.cat([A_main_values, A_bc_values], dim=0)
    global_indices = torch.stack([global_rows, global_cols], dim=0)

    # Build the global sparse operator of shape (n_edges*n_comp, n_cells*n_comp).
    A_all = torch.sparse_coo_tensor(global_indices, global_vals,
                                    size=(n_edges * n_comp, n_cells * n_comp),
                                    device=device, dtype=A_main.dtype).coalesce()

    # --- Build the global offset vector b_all ---
    b_all = torch.zeros(n_edges * n_comp, device=device, dtype=b_bc.dtype)
    # For the boundary rows, compute the global row indices similarly.
    # Create a vector for the local rows in the boundary operator.
    r_bc = torch.arange(b_bc.numel(), device=device)
    j_bc_for_b = r_bc // n_comp  # which boundary edge block each entry belongs to
    c_bc_for_b = r_bc % n_comp  # component within the block

    global_b_rows = bc_edge_global_indices[j_bc_for_b] * n_comp + c_bc_for_b
    b_all[global_b_rows] = b_bc


    return A_all.to_sparse_csr(), b_all


def lift_sparse_matrix(A_old: torch.Tensor, n_comp: int):
    """
    Lift a sparse matrix so that it acts on a flattened multi-component vector.
        U_out = torch.sparse.mm(A_old, U)   # U_out has shape (M, n_comp)
        U_out = torch.sparse.mm(A_new, U.flatten()).reshape(M, n_comp)
    A_old : torch.sparse.Tensor
        A sparse matrix in COO format of shape (M, N).
    n_comp : int
        The number of components (i.e. the second dimension of U).
    A_new : torch.sparse.Tensor
        The "lifted" sparse matrix of shape (M*n_comp, N*n_comp) that operates on a flattened U.
    """
    # Get the original indices and values.
    # indices_old is a tensor of shape (2, nnz), where nnz is the number of nonzero entries.
    A_old = A_old.coalesce()

    indices_old = A_old.indices()  # shape: (2, nnz)
    values_old = A_old.values()  # shape: (nnz,)
    M, N = A_old.size()
    nnz = values_old.size(0)
    device = A_old.device

    # Create a vector for the component indices: 0, 1, ..., n_comp - 1.
    comp = torch.arange(n_comp, device=device)  # shape: (n_comp,)

    # For each nonzero entry in A_old, we replicate the index for each component.
    # The new row index for an entry originally at row i becomes:
    #    i_new = i * n_comp + c   for c in 0,..., n_comp-1.
    new_rows = indices_old[0].unsqueeze(1) * n_comp + comp.unsqueeze(0)  # shape: (nnz, n_comp)
    new_cols = indices_old[1].unsqueeze(1) * n_comp + comp.unsqueeze(0)  # shape: (nnz, n_comp)
    new_vals = values_old.unsqueeze(1).expand(nnz, n_comp)  # shape: (nnz, n_comp)

    # Flatten these arrays to create the COO indices for A_new.
    new_rows = new_rows.reshape(-1)  # shape: (nnz * n_comp,)
    new_cols = new_cols.reshape(-1)
    new_vals = new_vals.reshape(-1)

    new_indices = torch.stack([new_rows, new_cols], dim=0)  # shape: (2, nnz * n_comp)
    new_size = (M * n_comp, N * n_comp)

    # # For now, we don't need rho entries.
    # rho_mask = (new_indices[0] % 3 == 2)
    # new_indices = new_indices[:, ~rho_mask]
    # new_vals = new_vals[~rho_mask]

    A_new = torch.sparse_coo_tensor(new_indices, new_vals, size=new_size, device=device).coalesce()

    return A_new


def interleave_sparse_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Given sparse COO tensors a, b of shape (m, n), return sparse COO tensor
    of shape (2m, n) with rows interleaved:

        out[0] = a[0]
        out[1] = b[0]
        out[2] = a[1]
        out[3] = b[1]
        ...

    """
    if a.layout != torch.sparse_coo or b.layout != torch.sparse_coo:
        raise TypeError("a and b must be sparse COO tensors")

    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("a and b must both have shape (m, n)")

    a = a.coalesce()
    b = b.coalesce()

    m, n = a.shape

    ai = a.indices()
    bi = b.indices()

    # Map row r from a -> 2r, row r from b -> 2r + 1
    ai_new = ai.clone()
    bi_new = bi.clone()

    ai_new[0] = 2 * ai[0]
    bi_new[0] = 2 * bi[0] + 1

    indices = torch.cat([ai_new, bi_new], dim=1)
    values = torch.cat([a.values(), b.values()], dim=0)

    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(2 * m, n),
        device=a.device,
        dtype=a.dtype,
    ).coalesce()
