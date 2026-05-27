from cprint import c_print
import torch
import numpy as np

from mesh_gen.mesh_3d.meshes_fvm import gen_mesh_cube_sphere
from base_cfg import ARTEFACT_DIR
from time_fvm.mesh_utils.mesh_store import FacetBCTypes as E
from time_fvm.mesh_utils.mesh_store import Facet
from time_fvm.mesh_utils.fvm_mesh import FVMMesh3D
from time_fvm.fvm_equation import FVMEquation, FluidConstitution3D, FluidConstitution
from time_fvm.config_fvm import ConfigFVM
from time_fvm.config_fvm_3d import ConfigEllipse


def generate_mesh(cfg: ConfigFVM):
    c_print(f'Creating new mesh for {cfg.problem_setup}', "green")
    mesh_stuff = gen_mesh_cube_sphere(volume=[0.05, 0.1], cell_lnscale=cfg.lnscale)
    Xs, tet_idx, (int_edgs, bound_facet), facet_tag = mesh_stuff

    Xs = torch.from_numpy(Xs).float()
    tet_idx = torch.from_numpy(tet_idx).int()
    int_edgs, bound_facet = torch.from_numpy(int_edgs), torch.from_numpy(bound_facet)
    all_facet = torch.cat([int_edgs, bound_facet], dim=0)
    bc_facet_mask = torch.cat([torch.zeros_like(int_edgs[:, 0], dtype=torch.bool), torch.ones_like(bound_facet[:, 0], dtype=torch.bool)], dim=0)

    c_print(f'Number of mesh cells: {len(tet_idx)}', "green")
    c_print(f'Number of mesh edges: {len(all_facet)}', "green")

    return Xs, tet_idx, all_facet, bc_facet_mask, facet_tag, bound_facet


def init_conds_3D(mesh: FVMMesh3D, edge_tag, bound_edgs, phy_setup: FluidConstitution, cfg: ConfigEllipse):
    # Set initial conditions same as inlet
    inlet_cfg = cfg.inlet_cfg
    v_in = inlet_cfg.v_inf[0]
    T_in = inlet_cfg.T_inf
    rho_in = inlet_cfg.rho_inf

    # Boundary conditions
    bc_tags = {}
    for bc_idx, (e_tag, e_vert) in enumerate(zip(edge_tag, bound_edgs, strict=True)):
        if e_tag == "NavierWall":
            bc_tags[bc_idx] = Facet([E.Dirich, E.Dirich, E.Dirich, E.Neuman, E.Neuman], [0., 0, 0, None, None], [None, None, None, 0, 0], tag=e_tag)
        elif e_tag == "Farfield":
            bc_tags[bc_idx] = Facet([E.Inlet, E.Inlet, E.Inlet, E.Inlet, E.Inlet], tag=e_tag)
        elif e_tag == "Right":
            bc_tags[bc_idx] = Facet([E.Farfield, E.Farfield, E.Farfield, E.Farfield, E.Farfield], tag=e_tag)
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')

    # Initial conditions
    # centroids = mesh.centroids
    # x, y = centroids[:, 0], centroids[:, 1]
    n_cells = mesh.n_cells
    prims_init = torch.zeros([n_cells, 1]).repeat(1, 5)
    prims_init[:, 0] = v_in
    prims_init[:, 1] = 0
    prims_init[:, 2] = 0
    prims_init[:, 3] = rho_in
    prims_init[:, 4] = T_in

    V, rho, T = prims_init[:, :3], prims_init[:, 3:4], prims_init[:, 4:]

    momentum, rho, Q = phy_setup.primatives_to_state(V, rho, T)
    Us_init = torch.cat([momentum, rho, Q], dim=-1)
    return bc_tags, Us_init


def main():
    import pickle
    np.random.seed(1)
    torch.manual_seed(1)

    new_mesh = True

    cfg: ConfigFVM = ConfigEllipse()
    phy_setup = FluidConstitution3D(cfg, dim=3)

    if new_mesh:
        c_print(f'Generating new mesh...', "green")
        prob_definition = generate_mesh(cfg)
        Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = prob_definition
        mesh = FVMMesh3D(Xs, tri_idx, all_edgs, bc_edge_mask, device=cfg.device)
        pickle.dump({'mesh': mesh, "edge_tag": edge_tag, "bound_edgs": bound_edgs}, open(f"{ARTEFACT_DIR}/fvm_mesh_3d.pkl", "wb"))
    else:
        c_print(f'Loading mesh', "green")
        save_dict = pickle.load(open(f"{ARTEFACT_DIR}/fvm_mesh_3d.pkl", "rb"))
        mesh: FVMMesh3D = save_dict['mesh']
        edge_tag = save_dict['edge_tag']
        bound_edgs = save_dict['bound_edgs']

    print(f'{mesh.volumes.min() = }')

    # Set up initial conditions.
    if cfg.problem_setup == "ellipse":
        bc_tags, us_init = init_conds_3D(mesh, edge_tag, bound_edgs, phy_setup, cfg)
    else:
        raise ValueError(f'Unknown mode {cfg.problem_setup}')

    solver = FVMEquation(cfg, phy_setup, mesh, cfg.N_comp, bc_tags, us_init=us_init)
    solver.solve()


if __name__ == "__main__":
    # PYTHONPATH=/home/maccyz/Documents/FVM_solver:/home/maccyz/Documents/FVM_solver/tetgen:/home/maccyz/Documents/FVM_solver/tetgen/src
    print("Running fvm ")
    print()
    main()
