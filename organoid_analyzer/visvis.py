import numpy as np
import imageio

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from . import morpho

def show_snakes(img, *snakes):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    
    for snake, color in zip(snakes, 'rgbcmy'):
        if isinstance(snake, (list, tuple)):
            snake = morpho.snake_from_extent(snake, img.shape)
        ax.plot(snake[:, 0], snake[:, 1], '--' + color, lw=3)
        
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])


from matplotlib import animation, rc
from IPython.display import HTML    
    
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
