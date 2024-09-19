from .utils import *
import pylab as plt
from skimage import morphology,feature

def interact_find_angles(X, aspect='auto',cmap = 'gray',clim = None,**kwargs):
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
                    aspect = aspect,
                    clim=clim,
                    cmap = cmap,
                    **kwargs)
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
        if not type(val) in [float,int,np.int64,np.float64,np.int32]:
            if val.key in ['right','d']:
                islide.set_val(np.clip(islide.val + 1,0,len(X)-1))
            elif val.key in ['left','a']:
                islide.set_val(np.clip(islide.val - 1,0,len(X)-1))
        iplane = int(np.floor(islide.val))
        im.set_data(X[iplane])
        if not type(val) in [float,int,np.int64,np.float64,np.int32]:
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

def interact_show_stack(X,cmap = 'gray',clim = None,**kwargs):
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
                    clim=clim,
                   cmap = cmap,
                   **kwargs)
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
        if not type(val) in [float,int,np.int64,np.float64,np.int32]:
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

def interact_stack_overlay_areas(X,atlas, cmap = 'gray_r',clim = None,overlay_color = 'darkred'):
    '''
    Interactive stack plot using matplotlib
       - left arrow or "a" key (decrease slice)
       - right arrow or "d" key (increase slice)

    Example:

       - Y is a SLICESxHxW array

    '''
    from matplotlib.widgets import Slider
    fig = plt.figure()
    fig.clf()
    ax = fig.add_axes([0,0,1,1])
    iframe = len(X)//2
    def get_edges(iframe):
        edg = feature.canny(atlas[:,:,iframe],sigma = 3)
        return morphology.dilation(edg).astype(bool)
    
    cm1 = plt.matplotlib.colors.ListedColormap(['none', overlay_color])

    im = plt.imshow(X[:,:,iframe],
                    aspect = 'auto',
                    clim=clim,
                   cmap = cmap)
    ed = plt.imshow(get_edges(iframe),cmap = cm1) 

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
        if not type(val) in [float,int,np.int64,np.float64,np.int32]:
            if val.key in ['right','d']:
                islide.set_val(np.clip(islide.val + 1,0,len(X)-1))
            elif val.key in ['left','a']:
                islide.set_val(np.clip(islide.val - 1,0,len(X)-1))
        f = int(np.floor(islide.val))
        im.set_data(X[:,:,f])
        ed.set_data(get_edges(f))
        txt.set_text(f)
    islide.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', update)
    return


def interact_rotate(x,cmap = 'gray',clim = None,
                    steps = 2,
                    include_atlas_label_indication = True, # write names areas in the oriented atlas
                    **kwargs):
    ''' 
    Function to manually approximate the alignment of the raw stack to the reference atlas
    '''
    
    res = dict(angles = [0,0,0],
               flip_x = False,
               flip_y = False,
               points_view0 = [],
               points_view1 = [],
               points_view2 = [])
    
    fig = plt.figure(facecolor = 'k')
    ax = []
    ims = []
    
    iframe = [40,40,40]
    #x = downsample_stack(X,[0.3,0.3,0.3])
    
    xviews = [x.transpose(2,0,1),x.transpose(1,2,0),x]
    iframe = [i.shape[0]//2 for i in xviews]
    points = [[] for i in range(3)]
    points_plots = []
    fits = []
    angles_text = []
    if clim is None:
        clim = [np.percentile(x,10),np.percentile(x,99.5)]
        print(clim)
    for i in range(3):
        ax.append(fig.add_subplot(1,3,1+i))
    
        ims.append(ax[-1].imshow(xviews[i][iframe[i]],
                             clim=clim,
                             cmap = cmap,
                             **kwargs))
        points_plots.append(ax[-1].plot(np.nan,np.nan,'yo--',alpha=0.4)[0])
        fits.append(ax[-1].plot(np.nan,np.nan,'r-',alpha=0.4)[0])
        angles_text.append(ax[-1].text(0,-1,'',color = 'y'))
        plt.axis('off')
    if include_atlas_label_indication:
        ax[2].text(0,x.shape[1]//2,'cortex',color='orange',rotation=90,ha='center',va = 'center')
        ax[2].text(x.shape[0]//2,0,'cerebellum',color='orange',ha='center',va = 'center',fontsize = 9)

    def check_angles():
        for i in range(3):
            if len(points[i]):
                p = np.stack(points[i])
                res[f'points_view{i}'] = p
                if len(points[i])>1:
                    ft = np.polyfit(*p.T,1)
                    xx = [np.min(p[:,0]),np.max(p[:,0])]
                    yy = np.polyval(ft,xx)
                    res[f'xy_view{i}'] = np.stack([xx,yy])
                    x = xx[1]-xx[0]
                    y = yy[1]-yy[0]
                    if i == 0: # cos
                        res['angles'][i] = np.rad2deg(np.arcsin(y/np.sqrt(x**2 + y**2)))
                        if res['angles'][i] > 90:
                            res['angles'][i] = res['angles'][i] - 180
                    elif i == 1: # sin
                        res['angles'][i] = np.rad2deg(np.arccos(y/np.sqrt(x**2 + y**2)))
                        if res['angles'][i] > 90:
                            res['angles'][i] = res['angles'][i]- 180
                    elif i == 2:
                        print('Not implemented')

    def on_scroll(event):
        if event.inaxes is None:
            return
        increment = steps if event.button == 'up' else -steps
        idx = ax.index(event.inaxes)
        iframe[idx] += increment
        iframe[idx] = np.clip(iframe[idx],0,len(xviews[idx])-1)
        
        ims[idx].set_data(xviews[idx][iframe[idx]])
        ims[idx].figure.canvas.draw()
        
    def on_click(event):
        if event.inaxes is None:
            return
        idx = ax.index(event.inaxes)
        if idx == 2: # then check if you have to flip the axis
            if event.button == 1:
                res['flip_x'] = not res['flip_x']
                xviews[2] = xviews[2][:,:,::-1]
            elif event.button == 3:
                res['flip_y'] = not res['flip_y']
                xviews[2] = xviews[2][:,::-1,:]
            ims[idx].set_data(xviews[idx][iframe[idx]])
        else:
            if event.button == 1:
                points[idx].append([event.xdata,event.ydata])
                p = np.stack(points[idx])
                points_plots[idx].set_xdata(p[:,0])
                points_plots[idx].set_ydata(p[:,1])
                if len(points[idx])>1:
                    #compute the angles
                    check_angles()
                    fits[idx].set_xdata(res[f'xy_view{idx}'][0])
                    fits[idx].set_ydata(res[f'xy_view{idx}'][1])
                    angles_text[idx].set_text('{0:2.1f}deg'.format(res['angles'][idx]))
            if event.button == 3:
                points[idx].pop()
                if not len(points[idx]):
                    points_plots[idx].set_xdata(np.nan)   
                    points_plots[idx].set_ydata(np.nan)   
                else:
                    p = np.stack(points[idx])
                    points_plots[idx].set_xdata(p[:,0])
                    points_plots[idx].set_ydata(p[:,1])
        ims[idx].figure.canvas.draw()
        
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_release_event', on_click)
    plt.axis('off')
    plt.show(block=True)
    return res
