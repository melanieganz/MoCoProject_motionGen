def plot_grid(data, n_rows, n_cols, add_legend=True, figsize=(21,11), seed=1):
    """Plots random samples from a single dataset in a grid"""

    # Set seed
    random.seed(seed)

    # Set figure size
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    # Get data dimensions. Assumes data is 3-dimensional.
    rows, cols, dims = data.shape
 
    # Sample n_rows x n_cols random indices
    rand_samples = random.sample([i for i in range(rows)], n_rows * n_cols)
    indices = np.array(rand_samples).reshape((n_rows, n_cols))

    # Labels and colors. Assumes data is fMRI
    labels=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    colors=['red', 'green', 'blue', 'orange', 'teal', 'purple']

    # Create Plot
    for i in range(n_rows):
        for j in range(n_cols):
            for d in range(dims):
                axs[i,j].plot(data[indices[i,j], :, d], label=labels[d], color=colors[d])

    # Add legend
    if add_legend:
        legend = fig.legend(labels, loc='lower center', ncol=dims, bbox_to_anchor=(0.5, 0))
        for text in legend.get_texts():
            text.set_fontsize(14)
            text.set_weight('bold')
    
    # Plot figure
    plt.xlabel('Frames', fontsize=14, weight='bold')
    plt.ylabel('Millimeters (mm)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()


def plot_datasets(datasets, n_cols, add_legend=True, figsize=(21,11), seed=1):
    """Plots random samples from a number of datasets in a grid"""

    # Set seed
    random.seed(seed)

    # Create on row per dataset
    n_rows = len(datasets)
    
    # Set figure size
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    # Labels and colors. Assumes data is fMRI
    labels=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    colors=['red', 'green', 'blue', 'orange', 'teal', 'purple']
    
    # Create plot
    for i,dataset in enumerate(datasets):
        # Get data dimensions. Assumes data is 3-dimensional.
        rows, cols, dims = dataset.shape

        # Sample n_rows x n_cols random indices to plot
        rand_samples = random.sample([i for i in range(rows)], n_rows * n_cols)
        indices = np.array(rand_samples).reshape((n_rows, n_cols))
        
        for j in range(n_cols):
            for d in range(dims):
                axs[i,j].plot(dataset[indices[i,j], :, d], color=colors[d])
    
    # Add legend
    if add_legend:
        legend = fig.legend(labels, loc='lower center', ncol=dims, bbox_to_anchor=(0.5, 0))
        for text in legend.get_texts():
            text.set_fontsize(14)
            text.set_weight('bold')
    
    # Plot figure
    plt.xlabel('Frames', fontsize=14, weight='bold')
    plt.ylabel('Millimeters (mm)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()


def plot_column(data, n_rows, add_legend=True, figsize=(9, 7), seed=1):
    """Plots random samples from a single dataset in a column"""

    # Set random seed for replication purposes
    np.random.seed(seed)

    # Set figure size
    fig, axs = plt.subplots(n_rows, 1, figsize=figsize)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    # Get data dimensions. Assumes data is 3-dimensional.
    rows, cols, dims = data.shape
    
    # Labels and colors. Assumes data is fMRI
    labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    colors = ['red', 'green', 'blue', 'orange', 'teal', 'purple']

    # Create plot
    for i in range(n_rows):
        # Pick random index (with replacement)
        idx = np.random.randint(0, rows)

        for d in range(dims):
            axs[i].plot(data[idx, :, d], label=labels[d], color=colors[d])
    
    # Add legend
    if add_legend:
        legend = fig.legend(labels, loc='lower center', ncol=dims, bbox_to_anchor=(0.5, 0))
        for text in legend.get_texts():
            text.set_fontsize(14)
            text.set_weight('bold')
    
    # Plot figure
    plt.xlabel('Frames', fontsize=14, weight='bold')
    plt.ylabel('Millimeters (mm)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()


def plot_mean_timeseries(data, add_legend=True):
    """Plots the mean timeseries of a dataset"""
    
    # Set figure
    fig = plt.figure(figsize=(10,10))

    # Get data dimensions. Assumes data is 3-dimensional.
    rows, cols, dims = data.shape

    # Labels and colors. Assumes data is fMRI
    labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    colors = ['red', 'green', 'blue', 'orange', 'teal', 'purple']
    
    # Create plot
    for d in range(dims):
        mean_curve = [data[:, c, d].mean() for c in range(cols)]
        plt.plot(mean_curve, label=labels[d], color=colors[d])
    
    # Add legend
    if add_legend:
        legend = fig.legend(labels, loc='lower center', ncol=dims, bbox_to_anchor=(0.5, -0.05))
        for text in legend.get_texts():
            text.set_fontsize(14)
            text.set_weight('bold')
    
    # Plot figure
    plt.xlabel('Frames', fontsize=14, weight='bold')
    plt.ylabel('Millimeters (mm)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()