from webbrowser import get
from SyMBac.simulation import Simulation
from SyMBac.PSF import PSF_generator
from SyMBac.renderer import Renderer
from SyMBac.PSF import Camera
from SyMBac.lineage import Lineage
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import toml
from pathlib import Path
import zarr
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from skimage.exposure import rescale_intensity
import networkx as nx
import skimage


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
    # real_image = np.zeros((256, 46))
    filepath = "/groups/funke/home/sistaa/code/SyMBac/scripts/images/220510_bsub_degron_rap_BF_raw.tif"
    real_image = Image.open(filepath, "r")
    real_image = np.array(real_image)
    # print(real_image.size)
    renderer = Renderer(
        simulation=simulation, PSF=psf, real_image=real_image, camera=camera
    )
    means = (
        image_config["media_mean"],
        image_config["cell_mean"],
        image_config["device_mean"],
    )
    means_array = np.array(means)
    variances = (
        image_config["media_var"],
        image_config["cell_var"],
        image_config["device_var"],
    )
    vars_array = np.array(variances)
    renderer.image_params = (*means, means_array, *variances, vars_array)
    renderer.params = renderer_config
    renderer.params["match_fourier"] = False
    renderer.params["match_histogram"] = True
    renderer.params["match_noise"] = True
    return renderer

def get_lineage_graph(simulation, id_offset):
    lineage_graph = Lineage(simulation).temporal_lineage_graph
    # this has nodes with (cell_id, time). We need to add max_id to get the new seg id for each frame
    # it also has edges between sisters, rather than mother/daughter at division
    div_nodes = [node for node, degree in lineage_graph.out_degree() if degree > 1]
    for div_node in div_nodes:
        cell_id, time = div_node
        for child in list(lineage_graph.successors(div_node)):
            child_id, child_time = child
            if time == child_time:
                lineage_graph.remove_edge(div_node, child)
                actual_parent = (cell_id, time -1)
                lineage_graph.add_edge(actual_parent, child)

    mapping = {}
    for node in lineage_graph.nodes():
        old_mask_id, time = node
        new_mask_id = old_mask_id + id_offset[time]
        mapping[node] = new_mask_id

    return nx.relabel_nodes(lineage_graph, mapping)

def generate_video_sample(
    renderer,
    params,
    save_dir,
    output_zarr,
    output_group,
    media_multiplier,
    cell_multiplier,
    device_multiplier,
    sigma,
    burn_in,
    num_scenes,
    match_histogram,
    match_noise,
    match_fourier,
    mask_dtype=np.uint64,
):
    zarr_group = zarr.open_group(zarr.DirectoryStore(Path(save_dir) / output_zarr, dimension_separator="/"), "w")
    
    image_ds = zarr_group.create_dataset(output_group, shape=(num_scenes, *renderer.real_image.shape), dtype=np.uint16) 
    mask_ds = zarr_group.create_dataset("mask", shape=(num_scenes, *renderer.real_image.shape), dtype=mask_dtype) 
    max_id = 0
    id_offset = {} # time -> offset
    for scene_no in range(burn_in, burn_in + num_scenes):
        image, mask, _ = renderer.generate_test_comparison(
            media_multiplier=media_multiplier,
            cell_multiplier=cell_multiplier,
            device_multiplier=device_multiplier,
            sigma=sigma,
            match_fourier=match_fourier,
            match_histogram=match_histogram,
            match_noise=match_noise,
            debug_plot=False,
            noise_var=params["noise_var"],
            defocus=params["defocus"],
            halo_top_intensity=params["halo_top_intensity"],
            halo_bottom_intensity=params["halo_bottom_intensity"],
            halo_start=params["halo_start"],
            halo_end=params["halo_end"],
            random_real_image=None,
            scene_no=scene_no,
        )

        image = skimage.img_as_uint(rescale_intensity(image))
        image_ds[scene_no - burn_in] = image
        mask = mask.astype(mask_dtype)
        id_offset[scene_no] = max_id
        mask[mask > 0] += max_id
        max_id = np.max(mask)
        print(f"max_id in frame {scene_no} is {max_id}")
        mask_ds[scene_no - burn_in] = mask
    
    lineage_graph = get_lineage_graph(renderer.simulation, id_offset)
    with open(Path(output_zarr).with_suffix(".csv"), 'w') as f:
        f.write("id,parent_id\n")
        for node in lineage_graph.nodes():
            parents = list(lineage_graph.predecessors(node))
            assert len(parents) == 1
            parent = parents[0]
            f.write(f"{node},{parent}")
            
    

def generate_image_sample(
    renderer,
    params,
    save_dir,
    output_zarr,
    output_group,
    media_multiplier,
    cell_multiplier,
    device_multiplier,
    sigma,
    scene_no,
    match_histogram,
    match_noise,
    match_fourier,
    mask_dtype=np.uint16,
):

    image, mask, _ = renderer.generate_test_comparison(
        media_multiplier=media_multiplier,
        cell_multiplier=cell_multiplier,
        device_multiplier=device_multiplier,
        sigma=sigma,
        scene_no=scene_no,
        match_fourier=match_fourier,
        match_histogram=match_histogram,
        match_noise=match_noise,
        debug_plot=False,
        noise_var=params["noise_var"],
        defocus=params["defocus"],
        halo_top_intensity=params["halo_top_intensity"],
        halo_bottom_intensity=params["halo_bottom_intensity"],
        halo_start=params["halo_start"],
        halo_end=params["halo_end"],
        random_real_image=None,
    )

    image = skimage.img_as_uint(rescale_intensity(image))
    zarr_group = zarr.open_group(zarr.DirectoryStore(Path(save_dir) / output_zarr, dimension_separator="/"), "a", path = f"scene_{scene_no}")
    zarr_group[output_group] = image

    mask = mask.astype(mask_dtype)
    zarr_group["mask"] = mask


def generate_data_from_simulation(
    simulation,
    renderer,
    params,
    sample_amount,
    burn_in,
    n_jobs,
    n_samples,
    save_dir,
    output_zarr,  # just the filename
    output_group,
    in_series=False,
    mask_dtype=np.uint8,
):
    """
    Generates the training data from a Jupyter interactive output of generate_test_comparison

    Parameters
    ----------
    sample_amount : float
        The percentage sampling variance (drawn from a uniform distribution) to vary intensities by. For example, a
        sample_amount of 0.05 will randomly sample +/- 5% above and below the chosen intensity for cells,
        media and device. Can be used to create a little bit of variance in the final training data.
    burn_in : int
        Number of frames to wait before generating training data. Can be used to ignore the start of the simulation
        where the trench only has 1 cell in it.
    n_samples : int
        The number of training images to generate
    output_zarr : str
        The zarr in which to save the data
    in_series : bool
        Whether the images should be randomly sampled, or rendered in the order that the simulation was run in.
    seed : float
        Optional arg, if specified then the numpy random seed will be set for the rendering, allows reproducible rendering results.

    """

    if in_series:
        series_len = (simulation.sim_length) - burn_in
        n_series_to_sim = int(np.ceil(n_samples / series_len))

        media_multipliers = np.repeat(
            [
                np.random.uniform(1 - sample_amount, 1 + sample_amount)
                * params["media_multiplier"]
                for _ in range(n_series_to_sim)
            ],
            series_len,
        )
        cell_multipliers = np.repeat(
            [
                np.random.uniform(1 - sample_amount, 1 + sample_amount)
                * params["cell_multiplier"]
                for _ in range(n_series_to_sim)
            ],
            series_len,
        )
        device_multipliers = np.repeat(
            [
                np.random.uniform(1 - sample_amount, 1 + sample_amount)
                * params["device_multiplier"]
                for _ in range(n_series_to_sim)
            ],
            series_len,
        )
        sigmas = np.repeat(
            [
                np.random.uniform(1 - sample_amount, 1 + sample_amount)
                * params["sigma"]
                for _ in range(n_series_to_sim)
            ],
            series_len,
        )
       

    else:
        media_multipliers = [
            np.random.uniform(1 - sample_amount, 1 + sample_amount)
            * params["media_multiplier"]
            for _ in range(n_samples)
        ]
        cell_multipliers = [
            np.random.uniform(1 - sample_amount, 1 + sample_amount)
            * params["cell_multiplier"]
            for _ in range(n_samples)
        ]
        device_multipliers = [
            np.random.uniform(1 - sample_amount, 1 + sample_amount)
            * params["device_multiplier"]
            for _ in range(n_samples)
        ]
        sigmas = [
            np.random.uniform(1 - sample_amount, 1 + sample_amount) * params["sigma"]
            for _ in range(n_samples)
        ]
 

    generate_video_sample(
            renderer,
            params,
            save_dir,
            output_zarr,
            output_group,
            params["media_multiplier"],
            params["cell_multiplier"],
            params["device_multiplier"],
            params["sigma"],
            burn_in,
            n_samples,
            params["match_histogram"],
            params["match_noise"],
            params["match_fourier"],
            mask_dtype=mask_dtype,
    )


if __name__ == "__main__":
    config = toml.load("configs/default.toml")
    print("Config", config)
    simulation = make_simulation(config["simulation"], config["name"])
    phase_psf = make_psf(config["phase_psf"])
    camera = Camera(**config["camera"])
    burn_in = config["training_data"]["burn_in"]
    n_samples = config["training_data"]["n_samples"]
    

    phase_renderer: Renderer = make_renderer(
        config["phase_image_stats"], config["renderer"], simulation, phase_psf, camera
    )
    output_zarr = config["name"] + ".zarr"
    
    generate_data_from_simulation(
        simulation,
        phase_renderer,
        phase_renderer.params,
        output_zarr = output_zarr,  # just the filename
        output_group = "phase",
        mask_dtype=np.uint8,
        **config["training_data"],
    )

