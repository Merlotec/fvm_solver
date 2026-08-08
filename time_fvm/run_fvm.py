from cprint import c_print
import torch
import numpy as np

from mesh_gen.mesh_2d.meshes_fvm import gen_mesh_nozzle, gen_rand_mesh
from base_cfg import ARTEFACT_DIR
from time_fvm.mesh_utils.mesh_store import FacetBCTypes as E
from time_fvm.mesh_utils.mesh_store import Facet
from time_fvm.mesh_utils.fvm_mesh import FVMMesh2D
from time_fvm.fvm_equation import FVMEquation, FluidConstitution2D, FluidConstitution
from time_fvm.config_fvm import ConfigFVM, ConfigNozzle, ConfigEllipse


def generate_mesh(cfg: ConfigFVM):
    c_print(f'Creating new mesh for {cfg.problem_setup}', "green")
    if cfg.problem_setup == "nozzle":
        mesh_stuff = gen_mesh_nozzle(areas=[cfg.min_A, cfg.max_A], cell_lnscale=cfg.lnscale)
    elif cfg.problem_setup == "ellipse":
        mesh_stuff = gen_rand_mesh(areas=[cfg.min_A, cfg.max_A], cell_lnscale=cfg.lnscale,
                                   n_colliders=getattr(cfg, "n_colliders", 3))
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


def init_conds_nozzle(mesh: FVMMesh2D, edge_tag, bound_edgs, phy_setup: FluidConstitution, cfg: ConfigNozzle):
    # Initial conditions based on inlet inside engine, outlet outside.
    inlet_cfg = cfg.inlet_cfg
    T_in = inlet_cfg.T_inf
    rho_in = inlet_cfg.rho_inf

    exit_cfg = cfg.exit_cfg
    T_out = exit_cfg.T_inf
    rho_out = exit_cfg.rho_inf

    # Boundary conditions
    bc_tags = {}
    assert len(edge_tag) == len(bound_edgs), 'edge_tag/bound_edgs length mismatch'
    for bc_idx, (e_tag, e_vert) in enumerate(zip(edge_tag, bound_edgs)):
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


def wall_distance(mesh: FVMMesh2D, edge_tag, bound_edgs, wall_tag="NavierWall"):
    """Distance from every cell centroid to the nearest no-slip wall segment.

    Returns None when the domain has no walls.  Exact point-to-segment distance (not
    distance to segment midpoints), so it stays accurate for coarse collider surfaces.
    """
    is_wall = torch.tensor([t == wall_tag for t in edge_tag], dtype=torch.bool)
    if not bool(is_wall.any()):
        return None

    verts = mesh.vertices.to(torch.float32)
    segs = bound_edgs[is_wall].long()
    a, b = verts[segs[:, 0]], verts[segs[:, 1]]          # [S, 2] each
    ab = b - a
    ab2 = (ab * ab).sum(-1).clamp_min(1e-12)             # [S]

    # Chunk over cells: the full [n_cells, n_segments] matrix is too large to hold.
    out = []
    for chunk in mesh.centroids.to(torch.float32).split(4096):
        ap = chunk.unsqueeze(1) - a.unsqueeze(0)                     # [c, S, 2]
        t = ((ap * ab.unsqueeze(0)).sum(-1) / ab2).clamp(0.0, 1.0)   # [c, S]
        proj = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)
        out.append((chunk.unsqueeze(1) - proj).norm(dim=-1).min(dim=1).values)
    return torch.cat(out)


def init_conds_ellipses(mesh: FVMMesh2D, edge_tag, bound_edgs, phy_setup: FluidConstitution, cfg: ConfigEllipse):
    # Set initial conditions same as inlet
    inlet_cfg = cfg.inlet_cfg
    v_in = inlet_cfg.v_n_inf
    v_t = inlet_cfg.v_t_inf          # tangential inflow -> angled freestream (0 = axial)
    T_in = inlet_cfg.T_inf
    rho_in = inlet_cfg.rho_inf

    # Freestream as a global vector when given (arbitrary angle), else the legacy
    # axial/tangential pair.
    v_inf_vec = getattr(inlet_cfg, "v_inf", None)
    if v_inf_vec is not None:
        vx, vy = float(v_inf_vec[0]), float(v_inf_vec[1])
    else:
        vx, vy = float(v_in), float(v_t)

    # Boundary conditions
    bc_tags = {}
    assert len(edge_tag) == len(bound_edgs), 'edge_tag/bound_edgs length mismatch'
    for bc_idx, (e_tag, e_vert) in enumerate(zip(edge_tag, bound_edgs)):
        if e_tag == "NavierWall":
            bc_tags[bc_idx] = Facet([E.Dirich, E.Dirich, E.Neuman, E.Neuman], [0., 0, None, None], [None, None, 0, 0], tag=e_tag)
        elif e_tag == "Left":
            bc_tags[bc_idx] = Facet([E.Inlet, E.Inlet, E.Inlet, E.Inlet], [None, None, None, None], [None, None, None, None], tag=e_tag)
        elif e_tag in ("Right", "Farfield"):
            # "Farfield" is the whole outer boundary of a rotatable-freestream domain;
            # the characteristic BC picks inflow vs outflow per face from its normal.
            bc_tags[bc_idx] = Facet([E.Farfield, E.Farfield, E.Farfield, E.Farfield], [None, None, None, None], [None, None, None, None], tag=e_tag)
        else:
            raise ValueError(f'Unknown edge tag {e_tag}')

    # Initial conditions
    n_cells = mesh.n_cells
    prims_init = torch.zeros([n_cells, 1]).repeat(1, 4)

    # Blend the freestream to zero at collider surfaces.  A uniform field would
    # contradict the no-slip wall BC at t=0, impulsively halting the fluid touching each
    # collider and firing one outward acoustic pulse per obstacle.
    d = wall_distance(mesh, edge_tag, bound_edgs)
    if d is None:
        ramp = torch.ones(n_cells)
    else:
        h = float(mesh.volumes.mean().sqrt())    # representative cell size
        ramp = torch.tanh(d / (3.0 * h))         # 0 at the wall -> 1 in the freestream

    prims_init[:, 0] = vx * ramp
    prims_init[:, 1] = vy * ramp
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

    new_mesh = True

    cfg: ConfigFVM = ConfigEllipse()
    phy_setup = FluidConstitution2D(cfg, dim=2)

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
        c_print(f'Loaded mesh. {mesh.n_cells = }, {mesh.n_facets = }', "green")

    print(f'{mesh.volumes.min() = }')

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
