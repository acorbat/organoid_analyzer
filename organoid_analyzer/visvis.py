import imageio

from IPython.display import HTML
from matplotlib import animation, rc
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from . import morpho
from . import fluorescence_estimation as fe
from . import image_manager as im

# Define useful cmap
HiLo_cmap = plt.cm.get_cmap('Greys_r')
HiLo_cmap.set_under('b')
HiLo_cmap.set_over('r')


def show_snakes(img, *snakes):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    
    for snake, color in zip(snakes, 'rgbcmy'):
        if isinstance(snake, (list, tuple)):
            snake = morpho.snake_from_extent(snake, img.shape)
        ax.plot(snake[:, 0], snake[:, 1], '--' + color, lw=3)
        
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    
def animate_stack_snakes(stack, *stack_snakes):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    img = stack[0, :, :]
    axi = ax.imshow(img, cmap=plt.cm.gray)
    
    tmp = []
    for snakes in stack_snakes:
        itmp = []
        for snake in snakes:
            if isinstance(snake, (list, tuple)):
                snake = morpho.snake_from_extent(snake, img.shape)        
            itmp.append(snake)
        tmp.append(itmp)
            
    stack_snakes = tmp
    
    snakes = stack_snakes[0]
    
    axps = []
    for snake, color in zip(snakes, 'rgbcmy'):
        axp = ax.plot(snake[:, 0], snake[:, 1], '--' + color, lw=3)
        axps.append(axp[0])
        
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    
    # animation function. This is called sequentially
    def animate(i):
        axi.set_data(stack[i, :, :])
        for ndx, axp in enumerate(axps):
            axp.set_data(stack_snakes[i][ndx][:, 0], stack_snakes[i][ndx][:, 1])
                         
        return (axi, *axps)
    
    anim = animation.FuncAnimation(fig, animate, #init_func=init,
                                   frames=len(stack_snakes), interval=20, blit=True)
    

    return HTML(anim.to_html5_video())


def visualizer(image):
    not_selected = True
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})

    ini_z = image.shape[1]//2
    ini_t = image.shape[0]//2

    subplot = SubPlot(axs[0], image, z=ini_z, t=ini_t)

    callback = Index(subplot, image.shape[1]-1, image.shape[0]-1)
    callback.cur_z = ini_z
    callback.cur_t = ini_t

    plt.sca(axs[1])

    axs[1].axis('off')

    axprev_z = plt.axes([0.6, 0.05, 0.15, 0.075])
    axnext_z = plt.axes([0.75, 0.05, 0.15, 0.075])
    bnext_z = Button(axnext_z, 'Next Z')
    bnext_z.on_clicked(callback.next_z)
    bprev_z = Button(axprev_z, 'Previous Z')
    bprev_z.on_clicked(callback.prev_z)

    axprev_t = plt.axes([0.3, 0.05, 0.15, 0.075])
    axnext_t = plt.axes([0.45, 0.05, 0.15, 0.075])
    bnext_t = Button(axnext_t, 'Next T')
    bnext_t.on_clicked(callback.next_t)
    bprev_t = Button(axprev_t, 'Previous T')
    bprev_t.on_clicked(callback.prev_t)

    def chosen_t(event):
        plt.close()
        nonlocal not_selected
        not_selected = False

    axchoose = plt.axes([0.1, 0.05, 0.1, 0.075])
    bchoose = Button(axchoose, 'Select')
    bchoose.on_clicked(chosen_t)

    plt.show()

    while not_selected:
        continue

    return callback.cur_t


class SubPlot(object):

    def __init__(self, axs, stack, z=0, t=0):
        self.axs = axs
        self.stack = stack
        self.z = z
        self.t = t

        plt.sca(self.axs)

        self.img = self.axs.imshow(self.stack[self.t][self.z], cmap='Greys_r')

        self.set_title()
        plt.draw()

    def update(self):
        plt.sca(self.axs)
        self.img.set_data(self.stack[self.t][self.z])
        self.set_title()
        plt.draw()

    def set_title(self):
        self.axs.set_title('z = %s; t = %s' % (self.z, self.t))


class Index(object):

    def __init__(self, subplot, z_len, t_len):
        self.subplot = subplot

        self.cur_z = 0
        self.max_z = z_len

        self.cur_t = 0
        self.max_t = t_len

    def next_z(self, event):
        self.cur_z= min(self.cur_z + 1, self.max_z)

        self.subplot.z = self.cur_z
        self.subplot.update()

    def prev_z(self, event):
        self.cur_z = max(self.cur_z - 1, 0)

        self.subplot.z = self.cur_z
        self.subplot.update()

    def next_t(self, event):
        self.cur_t = min(self.cur_t + 1, self.max_t)

        self.subplot.t = self.cur_t
        self.subplot.update()

    def prev_t(self, event):
        self.cur_t = max(self.cur_t - 1, 0)

        self.subplot.t = self.cur_t
        self.subplot.update()


# Classifier plots


def plot_class_map(df, clf, ax):
    """Generates the classification map for the region contained in the df
    given, using the clf classifier.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing all the points that are to be plotted
    clf : classifier.CLassifier
        Trained classifier to generate the map
    ax : matplotlib.Axes
        Axes on which to plot
    """
    X = np.asarray(df[clf.cols])

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min*.95, x_max*1.05, 100),
                         np.linspace(y_min*.95, y_max*1.05, 100))
    Z = clf.clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # TODO: Should be made more general, for more classes
    Z = np.asarray(
        [1 if this == 'spherical' else 2 if this == 'normal' else 3 for this
         in Z])
    Z = Z.reshape(xx.shape)

    ax.pcolormesh(xx, yy, Z, cmap=cmap_light, vmin=1, vmax=3)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_xlabel(clf.cols[0].replace('_', ' '))
    ax.set_ylabel(clf.cols[1].replace('_', ' '))


def plot_trajectory(X, ints, ax):
    """Plots the trajectory of X with it's intensity ints on axis ax.

    Parameters
    ----------
    X : numpy.array (2D)
        Array containing the (x, y) positions of the trajectory
    ints : list, tuple, numpy.array (1D)
        Sequence of intensities of points of the trajectory
    ax : matplotlib.Axes
        Axes on which to make the plot
    """
    ax.plot(X[:, 0], X[:, 1], '#1f77b4', alpha=0.7)
    ss = [50] * len(ints)
    ss[-1] = 100
    ax.scatter(X[:, 0], X[:, 1], c=ints, s=ss, alpha=0.7, cmap='inferno')


def get_array_from_fig(fig):
    """Get's an array of the image to create a gif afterwards.

    Parameters
    ----------
    fig : matplotlib.Figure

    Returns
    -------
    numpy.array (Nx, Ny, 4) RGBH array of image
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    s, (width, height) = canvas.print_to_buffer()

    return np.fromstring(s, np.uint8).reshape((height, width, 4))


def make_gif(stacks, df, clf, save_dir):
    """Creates a gif at save_dir according to clf classification and using
    information at df alongside the stacks.

    Parameters
    ----------
    stacks : numpy.array (T, Nx, Ny)
        Time series of stacks to be displayed
    df : pandas.DataFrame
        DataFrame containing the information of everything to be plotted
    clf : classifier.Classifier
        Trained classifier to plot the map
    save_dir : path
        Path to where the gif is to be saved
    """
    if len(stacks) != len(df):
        raise ValueError('Stacks and DataFrame should be the same length')
    X = np.asarray(df[clf.cols])
    ints = np.asarray(df['fluo_estimation'])

    images = []

    for n in range(len(stacks)):
        fig = Figure(figsize=(15, 5), frameon=False)

        # Add and plot stack image

        ax = fig.add_subplot(121)
        ax.axis('off')

        ax.imshow(stacks[n], cmap='Greys_r')

        # Add and plot trajectory

        ax = fig.add_subplot(122)

        plot_class_map(df, clf, ax)
        plot_trajectory(X[:n+1], ints[:n+1], ax)

        img = get_array_from_fig(fig)
        images.append(img)

    imageio.mimsave(str(save_dir), images, fps=2)


def make_border_int_gif(this_df, paths, save_dir):
    this_file, fluo_file, auto_file = paths
    ts = this_df.timepoint.values
    zs = this_df.z.values
    max_coords_len = this_df.border_values.apply(len).max()
    keys = im.get_keys(this_file, last_time=ts[-1])[ts, zs]
    max_int = this_df.otsu_mean.max() + 3 * np.nanstd(this_df.otsu_mean.values)
    min_dist = this_df.equivalent_diameter.min() * 0.7 / 2
    max_dist = this_df.equivalent_diameter.max() * 1.3 / 2

    with imageio.get_writer(str(save_dir), mode='I', fps=2) as writer:
        for t in ts:
            print('Generating image for file %s and timepoint %s' % (str(this_file), t))
            df_t = this_df.query('timepoint == %s' % t)
            ints = df_t.border_values.values[0]
            sorted_points = np.asarray(df_t.sorted_coords.values[0]).T[:, ::-1]
            distance_points = df_t.border_distance.values[0]
            sorted_points[:, 0] -= 14
            sorted_points[:, 1] -= 12

            fig = plt.figure(figsize=(5, 7), dpi=200, constrained_layout=False)
            axs = []

            gs1 = fig.add_gridspec(nrows=2, ncols=1)
            axs.append(fig.add_subplot(gs1[0]))

            gs2 = gs1[1].subgridspec(nrows=2, ncols=1, hspace=0)
            axs.append(fig.add_subplot(gs2[0]))
            axs.append(fig.add_subplot(gs2[1], sharex=axs[1]))

            stack_yfp = im.load_image(fluo_file, keys[t])
            stack_cfpyfp = im.load_image(auto_file, keys[t])

            _, stack_yfp, corr_stack_cfpyfp = fe.correct_stacks(
                np.ones_like(stack_yfp), stack_yfp, stack_cfpyfp)

            this_im = axs[0].imshow(stack_yfp, cmap=HiLo_cmap, vmin=0,
                                    vmax=max_int)
            axs[0].scatter(sorted_points[:, 0], sorted_points[:, 1],
                           c=np.arange(len(sorted_points)), s=0.5)
            fig.colorbar(this_im, ax=axs[0])

            axs[0].axis('off')

            axs[1].scatter(np.arange(len(sorted_points)), ints,
                           c=np.arange(len(sorted_points)),
                           s=2)

            axs[2].scatter(np.arange(len(sorted_points)), distance_points,
                        c=np.arange(len(sorted_points)), s=0.5)
            axs[2].set_ylabel('Distance to centroid')
            axs[2].set_ylim((min_dist, max_dist))

            axs[1].set_ylim((0, max_int))
            axs[1].set_xlim((0, max_coords_len))
            axs[1].grid('on')
            axs[2].grid('on')
            axs[2].set_xlabel('pixel number')
            axs[1].set_ylabel('Fluorescence Estimation')
            axs[1].set_title('Timepoint: %s' % t)
            plt.tight_layout()
            img = get_array_from_fig(fig)
            writer.append_data(img)
            plt.close()


def make_segmentation_and_state_gif(this_df, paths, save_dir):
    this_file, fluo_file, auto_file = paths
    ts = this_df.timepoint.values
    zs = this_df.z.values
    keys = im.get_keys(this_file, last_time=ts[-1])[ts, zs]
    max_int = this_df.otsu_mean.max() + 3 * np.nanstd(this_df.otsu_mean.values)
    ints = this_df.total_otsu_mean.values

    with imageio.get_writer(str(save_dir), mode='I', fps=2) as writer:
        for t in ts:
            print('Generating image for file %s and timepoint %s' % (str(this_file), t))
            df_t = this_df.query('timepoint == %s' % t)
            state = df_t.state.values[0]
            snk = df_t.external_snake.values[0]
            sorted_points = morpho.sort_border(snk)

            fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=200)

            stack_tran = im.load_image(this_file, keys[t])
            stack_yfp = im.load_image(fluo_file, keys[t])
            stack_cfpyfp = im.load_image(auto_file, keys[t])

            stack_tran, stack_yfp, corr_stack_cfpyfp = fe.correct_stacks(
                stack_tran, stack_yfp, stack_cfpyfp)

            axs[0].imshow(stack_tran, cmap=HiLo_cmap)
            axs[0].scatter(sorted_points[:, 0], sorted_points[:, 1], s=0.25,
                           c='r')

            axs[0].axis('off')
            axs[0].set_title('Transmission')

            sorted_points[:, 0] -= 14
            sorted_points[:, 1] -= 12

            this_im = axs[1].imshow(stack_yfp, cmap=HiLo_cmap, vmin=0,
                                    vmax=max_int)
            fig.colorbar(this_im, ax=axs[1])

            axs[1].scatter(sorted_points[:, 0], sorted_points[:, 1], s=0.25,
                           c='r')

            axs[1].axis('off')
            axs[1].set_title('Fluorescence')

            axs[2].plot(ts, ints)
            axs[2].axvline(x=t, color='gray', ls='--', alpha=0.5)
            axs[2].set_xlabel('timepoint')
            axs[2].set_ylabel('Fluorescence Estimation')
            axs[2].set_title('Fluorescence Estimation')

            plt.suptitle('timepoint: %s, state: %s' % (t, state))
            plt.tight_layout()
            img = get_array_from_fig(fig)
            writer.append_data(img)
            plt.close()
