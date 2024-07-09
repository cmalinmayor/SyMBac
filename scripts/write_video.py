from SyMBac.simulation import Simulation
from SyMBac.PSF import PSF_generator
from SyMBac.renderer import Renderer
from SyMBac.PSF import Camera
import matplotlib.pyplot as plt
import toml
from pathlib import Path
import zarr
import numpy as np
from ipywidgets import interactive, fixed


def save_simulation_timeseries_zarr(scene, mask, outfile):
    root = zarr.open(outfile)
    root["mask"] = mask
    root["scene"] = scene


def plot_PSF(psf, figname):
    if "3d fluo" in psf.mode.lower():
        fig, axes = plt.subplots(1, 3)
        for dim, ax in enumerate(axes.flatten()):
            ax.axis("off")
            ax.imshow((psf.kernel.mean(axis=dim)), cmap="Greys_r")
            # scalebar = ScaleBar(self.scale, "um", length_fraction=0.3)
            # ax.add_artist(scalebar)
        plt.savefig(figname)
    else:
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(psf.kernel, cmap="Greys_r")
        # scalebar = ScaleBar(self.scale, "um", length_fraction=0.25)
        # ax.add_artist(scalebar)
        plt.savefig(figname)


def make_simulation(config, name):
    simulation = Simulation(**config)
    simulation.run_simulation(show_window=False)
    simulation.draw_simulation_OPL(do_transformation=False, label_masks=True)
    output_zarr_path = Path(name + ".zarr")
    save_simulation_timeseries_zarr(
        simulation.OPL_scenes, simulation.masks, output_zarr_path
    )
    return simulation


def make_psf(config):
    kernel = PSF_generator(**config)
    kernel.calculate_PSF()
    plot_PSF(kernel, f"kernel_{config['mode']}.png")
    return kernel


def make_renderer(image_config, renderer_config, simulation, psf, camera) -> Renderer:
    real_image = np.zeros((256, 46))
    renderer = Renderer(
        simulation=simulation, PSF=psf, real_image=real_image, camera=camera
    )
    means = (image_config["media_mean"], image_config["cell_mean"], image_config["device_mean"])
    means_array = np.array(means)
    variances = (image_config["media_var"], image_config["cell_var"], image_config["device_var"])
    vars_array = np.array(variances)
    renderer.image_params = (*means, means_array, *variances, vars_array)
    renderer.params = renderer_config
    renderer.params["match_fourier"] = False
    renderer.params["match_histogram"] = False
    renderer.params["match_noise"] = False
    return renderer


if __name__ == "__main__":
    config = toml.load("configs/default.toml")
    print("Config", config)
    simulation = make_simulation(config["simulation"], config["name"])
    phase_psf = make_psf(config["phase_psf"])
    fluo_psf = make_psf(config["fluo_psf"])
    camera = Camera(**config["camera"])
    # fluo_renderer: Renderer = make_renderer(
    #     config["fluo_image_stats"], config["renderer"], simulation, fluo_psf, camera
    # )
    # fluo_renderer.generate_training_data(**config["training_data"])

    phase_renderer: Renderer = make_renderer(
        config["phase_image_stats"], config["renderer"], simulation, fluo_psf, camera
    )
    phase_renderer.generate_training_data(**config["training_data"])
