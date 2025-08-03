import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb

def plot_frames_row(frames, wb, figlabel, caption):
    nrows = 1
    ncols = len(frames)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 2))
    # Handle single frame (single column) case
    if ncols == 1:
        axs = [axs]

    for col_i in range(ncols):
        axs[col_i].imshow(frames[col_i])
        axs[col_i].set_xticks([])
        axs[col_i].set_yticks([])

    axs[0].set_ylabel(caption)

    plt.tight_layout()
    wb.log({figlabel: fig})
    plt.close()


def plot_frames(frames, wb, figlabel, caption, ncols=5, save_as_pdf=False, save_individual_pngs=False, save_dir="tmp_outputs"):
    """
    Plots the given frames in multiple rows with a specified number of columns.

    Args:
        frames (list): List of frames to plot.
        wb (wandb): WandB instance for logging.
        figlabel (str): Label for the figure in WandB.
        caption (str): Caption for the plot.
        ncols (int): Number of columns per row.
    """
    nrows = (len(frames) + ncols - 1) // ncols  # Calculate the number of rows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))

    # Flatten axs for easier indexing if nrows > 1
    axs = axs.flatten() if nrows > 1 else [axs]

    for i, frame in enumerate(frames):
        axs[i].imshow(frame)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        if save_individual_pngs:
            # Save individual frame as PNG and log to wandb
            tmp_fname = f"{save_dir}/{figlabel}_frame_{i}.png"
            plt.imsave(tmp_fname, frame)
            if wb is not None:
                wb.log({f"{figlabel}_frame_{i}": wandb.Image(tmp_fname)})

    # Hide unused axes
    for i in range(len(frames), len(axs)):
        axs[i].axis("off")

    axs[0].set_ylabel(caption)

    plt.tight_layout()
    if wb is not None:
        wb.log({figlabel:fig})

    if save_as_pdf:
        tmp_fname = "{}/{}.pdf".format(save_dir, figlabel)
        plt.savefig(tmp_fname, dpi=300, bbox_inches='tight')
        if wb is not None:
            wb.save(tmp_fname)

    plt.close()


def plot_logs(wandb, output_log):
    timesteps = len(output_log["t"])
    for key in output_log.keys():
        if key == "t":
            continue

        nrows = timesteps
        ncols = len(output_log["z"][0])
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 31))

        frames = output_log[key]

        # Handle single row/column cases
        if nrows == 1 and ncols == 1:
            axs = np.array([[axs]])
        elif nrows == 1:
            axs = np.array([axs])
        elif ncols == 1:
            axs = np.array([[ax] for ax in axs])

        for row_i in range(nrows):
            for col_i in range(ncols):
                axs[row_i][col_i].imshow(frames[row_i][col_i])
                axs[row_i][col_i].set_xticks([])
                axs[row_i][col_i].set_yticks([])           
            axs[row_i][0].set_ylabel("t = {}".format(output_log["t"][row_i]))

        plt.tight_layout()
        wandb.log({key+"_log": fig})


def seva_tensor_to_np_plottable(seva_tensor):
    """
    (B, C, H, W) tensor in range [-1, 1] to (B, H, W, C) ndarray of int8 in range [0, 255]
    """
    return (((seva_tensor.permute((0,2,3,1)) + 1) / 2.0)* 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

def np_plottable_to_seva_tensor(arr):
    return ((torch.tensor(arr, dtype=torch.float32).permute((0, 3, 1, 2)) / 255.0) * 2.0) - 1.0