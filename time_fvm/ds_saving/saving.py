import numpy as np
from datetime import datetime
import os
import torch
from cprint import c_print

from time_fvm.utils.plotting import plot_interp_cell
from time_fvm.fvm_stepping.facet_process import FacetFlux
from base_cfg import ARTEFACT_DIR

class Saver:
    def __init__(self, E_props: FacetFlux, save_dir=None):
        if save_dir is None:
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            self.save_dir = f'{ARTEFACT_DIR}/fvm_saves/{timestamp}'
        else:
            self.save_dir = save_dir

        c_print(f'Saving results to {self.save_dir}', color="bright_cyan")
        os.makedirs(self.save_dir, exist_ok=True)

        # Save mesh properties
        mesh = E_props.mesh
        fvm = mesh.fvm_mesh
        # Mesh property: main
        centroids = fvm.centroids.cpu().numpy()  # shape = [n_cells, 2]
        triangles = fvm.cells.cpu().numpy()  # shape = [n_cells, 3]
        vertices = fvm.vertices.cpu().numpy()    # shape = [n_vertices, 2]
        edges = fvm.facets.cpu().numpy()          # shape = [n_edges, 2]
        # Mesh property: BC edges
        bc_edge_mask = mesh.bc_facet_mask.cpu().numpy()                # shape = [n_edges]
        bc_cpu_mask = fvm.bc_facet_mask                      # CPU boolean mask for indexing CPU tensors
        bc_midpoints = fvm.midpoints[bc_cpu_mask].numpy()     # shape = [n_bc_edge, 2]
        bc_normals = fvm.normals[bc_cpu_mask].numpy()         # shape = [n_bc_edge, 2]
        bc_type_str = E_props.bc_type_str                       # shape = [n_bc_edge]
        mesh_props = {"triangles": triangles, "vertices": vertices, "centroids": centroids, "edges": edges,
                      "bc_midpoints": bc_midpoints, "bc_normals": bc_normals,
                      "bc_edge_masK": bc_edge_mask, "bc_type_str": bc_type_str}
        np.savez_compressed(f'{self.save_dir}/mesh_props.npz', **mesh_props)
        c_print(f"Saved mesh properties to '{self.save_dir}/mesh_props.npz'", color="green")

    def save(self, t, E_props: FacetFlux, primatives):
        bc_facet_mask = E_props.mesh.fvm_mesh.bc_facet_mask
        # Save centroid values
        primatives = primatives  # shape = [n_cells, comp=4]
        # Save boundary edge values
        Vs_bc = E_props.Vs_facet[bc_facet_mask].mean(dim=1)  # shape = [n_bc_edge, comp=2]
        rho_bc = E_props.rho_facet[bc_facet_mask].mean(dim=1)  # shape = [n_bc_edge, comp=1]
        T_bc = E_props.T_facet[bc_facet_mask].mean(dim=1)  # shape = [n_bc_edge, comp=1]
        bc_vals = torch.cat([Vs_bc, rho_bc, T_bc], dim=1)  # shape = [n_bc_edge, comp=4]

        # Save to file.
        # Normalise the primatives and bc_vals and save as fp16
        prim_mean, prim_std = primatives.mean(dim=0, keepdim=True), primatives.std(dim=0, keepdim=True)
        prim_scale = (primatives - prim_mean) / prim_std
        bc_scale = (bc_vals - prim_mean) / prim_std
        values = {"t": t.cpu().item(),
                    "prim_mean": prim_mean.cpu().numpy(), "prim_std": prim_std.cpu().numpy(),
                    "cell_primatives": prim_scale.cpu().half().numpy(), "bc_primatives": bc_scale.cpu().half().numpy()}

        t = t.cpu().item()
        t_str = f'{t:.4g}'  # Ensure step number is zero-padded to 5 digits
        np.savez_compressed(f'{self.save_dir}/t_{t_str}.npz', **values)


def load_step(file_path):
    """
    Loads a single time step file and returns the un-normalized primative values.
    """
    data = np.load(file_path)
    prim_mean = data['prim_mean']
    prim_std = data['prim_std']
    cell_primatives_scaled = data['cell_primatives'].astype(np.float32)
    bc_primatives_scaled = data['bc_primatives'].astype(np.float32)

    cell_primatives = cell_primatives_scaled * prim_std + prim_mean
    bc_primatives = bc_primatives_scaled * prim_std + prim_mean

    return data['t'], torch.from_numpy(cell_primatives).float(), torch.from_numpy(bc_primatives).float()


def main(save_dir='/home/maccyz/Documents/FVM_solver/artefacts/fvm_saves/03-29_06-20-55'):
    """
    Plot out the saved mesh and time step data.
    """
    print(f"\nLoading from '{save_dir}'...")

    # Load mesh properties
    mesh_props_path = os.path.join(save_dir, 'mesh_props.npz')
    mesh_props = np.load(mesh_props_path)
    mesh_props = dict(mesh_props)
    mesh_props.pop("bc_type_str", None)
    # print(mesh_props)
    mesh_props = {k: torch.from_numpy(v) for k, v in mesh_props.items()}
    print(f'{mesh_props.keys() = }')

    # Find and load time-step files
    time_files = sorted([f for f in os.listdir(save_dir) if f.startswith('t_') and f.endswith('.npz')])
    if not time_files:
        print("No time-step files found.")
        return

    print(f"Found {len(time_files)} time-step file(s)")

    # Load and process the first time step file
    # Sort time files by time value
    time_files.sort(key=lambda x: float(x.split('_')[1].replace('.npz', '')))
    for save_i in time_files:
        t = save_i.split('_')[1].replace('.npz', '')
        file_path = os.path.join(save_dir, save_i)
        t, cell_primatives, bc_primatives = load_step(file_path)

        print(f"Time: {t:.4g}")
        plot_interp_cell(mesh_props['vertices'], cell_primatives.T[:2], mesh_props['triangles'], title=f't={t:.3g}')


if __name__ == '__main__':
    main()
