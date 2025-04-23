import matplotlib.pyplot as plt

def plot_frames_row(frames, wb, figlabel, caption):
    nrows = 1
    ncols = len(frames)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 2))
    for col_i in range(ncols):
        axs[col_i].imshow(frames[col_i])
        axs[col_i].set_xticks([])
        axs[col_i].set_yticks([])

    axs[0].set_ylabel(caption)

    plt.tight_layout()
    wb.log({figlabel: fig})
    plt.close()

def plot_logs(wandb, output_log):
    timesteps = len(output_log["t"])
    for key in output_log.keys():
        if key == "t":
            continue

        nrows = timesteps
        ncols = 21
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(31, 21))

        frames = output_log[key]

        for row_i in range(nrows):
            for col_i in range(ncols):
                axs[row_i][col_i].imshow(frames[row_i][col_i])
                axs[row_i][col_i].set_xticks([])
                axs[row_i][col_i].set_yticks([])           
            axs[row_i][0].set_ylabel("t = {}".format(output_log["t"][row_i]))

        plt.tight_layout()
        wandb.log({key+"_log": fig})