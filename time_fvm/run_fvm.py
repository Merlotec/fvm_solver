from cprint import c_print
import torch
import numpy as np

from mesh_gen.mesh_2d.meshes_fvm import gen_mesh_nozzle, gen_rand_mesh
from base_cfg import ARTEFACT_DIR
from time_fvm.mesh_utils.mesh_store import FacetBCTypes as E
from time_fvm.mesh_utils.mesh_store import Facet
from time_fvm.mesh_utils.fvm_mesh import FVMMesh2D
from time_fvm.fvm_equation import FVMEquation, PhysicalSetup
from time_fvm.config_fvm import ConfigFVM, ConfigNozzle, ConfigEllipse


def generate_mesh(cfg: ConfigFVM):
    c_print(f'Creating new mesh for {cfg.problem_setup}', "green")
    if cfg.problem_setup == "nozzle":
        mesh_stuff = gen_mesh_nozzle(areas=[cfg.min_A, cfg.max_A], cell_lnscale=cfg.lnscale)
    elif cfg.problem_setup == "ellipse":
        mesh_stuff = gen_rand_mesh(areas=[cfg.min_A, cfg.max_A], cell_lnscale=cfg.lnscale)
    else:
        raise ValueError(f'Unknown mode {cfg.problem_setup}')

    Xs, tri_idx, (int_edgs, bound_edgs), edge_tag = mesh_stuff

    Xs = torch.from_numpy(Xs).float()
    tri_idx = torch.from_numpy(tri_idx).int()
    int_edgs, bound_edgs = torch.from_numpy(int_edgs), torch.from_numpy(bound_edgs)
    all_edgs = torch.cat([int_edgs, bound_edgs], dim=0)
    bc_edge_mask = torch.cat([torch.zeros_like(int_edgs[:, 0], dtype=torch.bool), torch.ones_like(bound_edgs[:, 0], dtype=torch.bool)], dim=0)

    c_print(f'Number of mesh cells: {len(tri_idx)}', "green")
    c_print(f'Number of mesh edges: {len(all_edgs)}', "green")

    return Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs


def init_conds_nozzle(mesh: FVMMesh2D, edge_tag, bound_edgs, phy_setup: PhysicalSetup, cfg: ConfigNozzle):
    # Initial conditions based on inlet inside engine, outlet outside.
    inlet_cfg = cfg.inlet_cfg
    T_in = inlet_cfg.T_inf
    rho_in = inlet_cfg.rho_inf

    exit_cfg = cfg.exit_cfg
    T_out = exit_cfg.T_inf
    rho_out = exit_cfg.rho_inf

    # Boundary conditions
    bc_tags = {}
    for bc_idx, (e_tag, e_vert) in enumerate(zip(edge_tag, bound_edgs, strict=True)):
        if e_tag == "NavierWall":
            bc_tags[bc_idx] = Facet([E.Dirich, E.Dirich, E.Neuman, E.Neuman], [0., 0, None, None], [None, None, 0, 0])
        elif e_tag == "Side":
            bc_tags[bc_idx] = Facet([E.Farfield, E.Farfield, E.Farfield, E.Farfield], [None, None, None, None], [None, None, None, None])
        elif e_tag == "Left":
            # bc_tags[bc_idx] = Edge([E.Neuman, E.Dirich, E.Dirich, E.Dirich], [None, 0, rho_in, T_in], [0, None, None, None])
            bc_tags[bc_idx] = Facet([E.Inlet, E.Inlet, E.Inlet, E.Inlet], [None, None, None, None], [None, None, None, None], tag=e_tag)
        elif e_tag == "Right":
            bc_tags[bc_idx] = Facet([E.Farfield, E.Farfield, E.Farfield, E.Farfield], [None, None, None, None], [None, None, None, None])
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')

    # Initial conditions
    centroids = mesh.centroids
    x, y = centroids[:, 0], centroids[:, 1]
    n_cells = mesh.n_cells
    prims_init = torch.zeros([n_cells, 1]).repeat(1, 4)
    prims_init[:, 0] = 0
    prims_init[:, 1] = 0
    prims_init[:, 2] = rho_in * (x < .4) + rho_out * (x > .4)
    prims_init[:, 3] = T_in * (x < .4) + T_out * (x > .4)

    V, rho, T = prims_init[:, :2], prims_init[:, 2:3], prims_init[:, 3:]

    momentum, rho, Q = phy_setup.primatives_to_state(V, rho, T)
    Us_init = torch.cat([momentum, rho, Q], dim=-1)
    return bc_tags, Us_init


def init_conds_ellipses(mesh: FVMMesh2D, edge_tag, bound_edgs, phy_setup: PhysicalSetup, cfg: ConfigEllipse):
    # Set initial conditions same as inlet
    inlet_cfg = cfg.inlet_cfg
    v_in = inlet_cfg.v_n_inf
    T_in = inlet_cfg.T_inf
    rho_in = inlet_cfg.rho_inf

    # Boundary conditions
    bc_tags = {}
    for bc_idx, (e_tag, e_vert) in enumerate(zip(edge_tag, bound_edgs, strict=True)):
        if e_tag == "NavierWall":
            bc_tags[bc_idx] = Facet([E.Dirich, E.Dirich, E.Neuman, E.Neuman], [0., 0, None, None], [None, None, 0, 0], tag=e_tag)
        elif e_tag == "Left":
            bc_tags[bc_idx] = Facet([E.Inlet, E.Inlet, E.Inlet, E.Inlet], [None, None, None, None], [None, None, None, None], tag=e_tag)
        elif e_tag == "Right":
            bc_tags[bc_idx] = Facet([E.Farfield, E.Farfield, E.Farfield, E.Farfield], [None, None, None, None], [None, None, None, None], tag=e_tag)
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')

    # Initial conditions
    # centroids = mesh.centroids
    # x, y = centroids[:, 0], centroids[:, 1]
    n_cells = mesh.n_cells
    prims_init = torch.zeros([n_cells, 1]).repeat(1, 4)
    prims_init[:, 0] = v_in
    prims_init[:, 1] = 0
    prims_init[:, 2] = rho_in
    prims_init[:, 3] = T_in

    V, rho, T = prims_init[:, :2], prims_init[:, 2:3], prims_init[:, 3:]

    momentum, rho, Q = phy_setup.primatives_to_state(V, rho, T)
    Us_init = torch.cat([momentum, rho, Q], dim=-1)
    return bc_tags, Us_init


def main():
    import pickle
    np.random.seed(1)
    torch.manual_seed(1)

    new_mesh = False

    cfg: ConfigFVM = ConfigEllipse()
    phy_setup = PhysicalSetup(cfg)

    if new_mesh:
        c_print(f'Generating new mesh...', "green")
        prob_definition = generate_mesh(cfg)
        Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = prob_definition
        mesh = FVMMesh2D(Xs, tri_idx, all_edgs, bc_edge_mask, device=cfg.device)
        pickle.dump({'mesh': mesh, "edge_tag": edge_tag, "bound_edgs": bound_edgs}, open(f"{ARTEFACT_DIR}/fvm_mesh.pkl", "wb"))
    else:
        c_print(f'Loading mesh', "green")
        save_dict = pickle.load(open(f"{ARTEFACT_DIR}/fvm_mesh.pkl", "rb"))
        mesh: FVMMesh2D = save_dict['mesh']
        edge_tag = save_dict['edge_tag']
        bound_edgs = save_dict['bound_edgs']

    print(f'{mesh.areas.min() = }')

    # Set up initial conditions.
    if cfg.problem_setup == "ellipse":
        bc_tags, us_init = init_conds_ellipses(mesh, edge_tag, bound_edgs, phy_setup, cfg)
    elif cfg.problem_setup == "nozzle":
        bc_tags, us_init = init_conds_nozzle(mesh, edge_tag, bound_edgs, phy_setup, cfg)
    else:
        raise ValueError(f'Unknown mode {cfg.problem_setup}')

    solver = FVMEquation(cfg, phy_setup, mesh, cfg.N_comp, bc_tags, us_init=us_init)
    solver.solve()


if __name__ == "__main__":
    # PYTHONPATH=/home/maccyz/Documents/FVM_solver:/home/maccyz/Documents/FVM_solver/tetgen:/home/maccyz/Documents/FVM_solver/tetgen/src
    print("Running fvm ")
    print()
    main()
