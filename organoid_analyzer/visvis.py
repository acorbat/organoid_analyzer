import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button

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
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})

    subplot = SubPlot(axs[0], image, z=image.shape[1]//2, t=image.shape[0]//2)

    callback = Index(subplot, image.shape[1]-1, image.shape[0]-1)
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
        return callback.cur_t

    axchoose = plt.axes([0.1, 0.05, 0.1, 0.075])
    bchoose = Button(axchoose, 'Select')
    bchoose.on_clicked(chosen_t)

    plt.show()


class SubPlot(object):

    def __init__(self, axs, stack, z=0, t=0):
        self.axs = axs
        self.stack = stack
        self.z = z
        self.t = t

        plt.sca(self.axs)

        self.img = self.axs.imshow(self.stack[self.t][self.z])

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

