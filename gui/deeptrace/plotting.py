from .utils import *
import pylab as plt

def interact_find_angles(X,cmap = 'gray',clim = None):
    '''
    Interactive tool to find rotation angles using matplotlib
       - left arrow or "a" key (decrease slice)
       - right arrow or "d" key (increase slice)
       - x or X(add a point in the same plane for an x rotation)
       - y or Y (add a point in the same plane for an x rotation)
       - z or Z (add a point in the same plane for an x rotation)
    
Usage:

    1. Start the interactive tool: res = interact_find_angles(stack.downsampled_data[0]);
    2. Find a point in the image where there is a blood vessel
    3. hoover the mouse over it and press "z" (z1 will appear over that point)
    4. find another point on another slice that is also in the same plane.
    5. hoover the mouse over it and press "shift + z" (z2 will appear and the angle is computed)
    6. Select other points for "x" and/or "y" rotations
    7. rotated_stack = rotate_stack(stack.downsampled_data[0],
                                    *res['angles'])

    '''
    from matplotlib.widgets import Slider
    points_dict = dict(x = np.zeros((2,3),dtype=np.float32),
                       y = np.zeros((2,3),dtype=np.float32),
                       z = np.zeros((2,3),dtype=np.float32),
                       angles = [0,0,0])
    def reset_points():
        for k in ['x','y','z']:
            points_dict[k][:] = np.nan
    fig = plt.figure()
    fig.clf()
    ax = fig.add_axes([0,0,1,1])
    iframe = len(X)//2
    im = plt.imshow(X[iframe],
                    aspect = 'auto',
                    clim=clim,
                   cmap = cmap)
    txt = plt.text(0,0,iframe,color = 'w',va = 'top')
    points = dict()
    for k in ['x','y','z']:
        points[k] = [plt.text(-10,-10,k+'1',color = 'y',va = 'center',ha='center'),
                     plt.text(-10,-10,k+'2',color = 'y',va = 'center',ha='center')]

    reset_points()
    sliderax = fig.add_axes([0.01, 0.01, 0.15, 0.02])
    islide = Slider(sliderax,
                    'slice #',
                    valmin=0,
                    valmax=len(X)-1,
                    valstep=1,
                    initcolor='w',
                    valinit = len(X)//2)
    def update(val):
        if not type(val) in [float,int,np.int64,np.float64]:
            if val.key in ['right','d']:
                islide.set_val(np.clip(islide.val + 1,0,len(X)-1))
            elif val.key in ['left','a']:
                islide.set_val(np.clip(islide.val - 1,0,len(X)-1))
        iplane = int(np.floor(islide.val))
        im.set_data(X[iplane])
        if not type(val) in [float,int,np.int64,np.float64]:
            if val.key in ['right','d']:
                islide.set_val(np.clip(islide.val + 1,0,len(X)-1))
            elif val.key in ['left','a']:
                islide.set_val(np.clip(islide.val - 1,0,len(X)-1))
            elif val.key.lower() in ['x','y','z']:
                ii = 0
                k = val.key.lower()
                if val.key.isupper(): # z2 then 
                    ii = 1
                points_dict[k][ii] = [np.floor(val.xdata),np.floor(val.ydata),iplane]
            elif val.key in ['r','R']:
                # reset
                reset_points()
        # plot the points if in the correct plane
        for j,k in enumerate(['x','y','z']):
            for ii in range(2):
                if points_dict[k][ii,2] == np.float32(iplane):
                    points[k][ii].set_x(points_dict[k][ii,0])
                    points[k][ii].set_y(points_dict[k][ii,1])
                else:
                    points[k][ii].set_x(-10)
                    points[k][ii].set_y(-10)
        z = points_dict['z']
        if np.any(z != np.nan):
            points_dict['angles'][0] = np.rad2deg(np.arctan((z[0,2]-z[1,2])/(z[0,1]-z[1,1])))
        # set the plane number
        txt.set_text(iplane)
        fig.canvas.draw_idle()
    islide.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', update)
    return points_dict

def interact_show_stack(X,cmap = 'gray',clim = None):
    '''
    Interactive stack plot using matplotlib
       - left arrow or "a" key (decrease slice)
       - right arrow or "d" key (increase slice)

    Example:

    interact_show_stack(Y)

       - Y is a SLICESxHxW array

    '''
    from matplotlib.widgets import Slider
    fig = plt.figure()
    fig.clf()
    ax = fig.add_axes([0,0,1,1])
    iframe = len(X)//2
    im = plt.imshow(X[iframe],
                    aspect = 'auto',
                    clim=clim,
                   cmap = cmap)
    txt = plt.text(0,0,iframe,color = 'w',va = 'top')
    sliderax = fig.add_axes([0.01, 0.01, 0.15, 0.01])
    islide = Slider(sliderax,
                    'slice #',
                    valmin=0,
                    valmax=len(X)-1,
                    valstep=1,
                    initcolor='w',
                    valinit = len(X)//2)
    def update(val):
        if not type(val) in [float,int,np.int64,np.float64]:
            if val.key in ['right','d']:
                islide.set_val(np.clip(islide.val + 1,0,len(X)-1))
            elif val.key in ['left','a']:
                islide.set_val(np.clip(islide.val - 1,0,len(X)-1))
        f = int(np.floor(islide.val))
        im.set_data(X[f])
        txt.set_text(f)
    islide.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', update)
    return 
