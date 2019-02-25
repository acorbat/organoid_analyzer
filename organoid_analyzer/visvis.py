import matplotlib.pyplot as plt

from organoid_analyzer import morpho

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


