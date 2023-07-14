# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
##############################################################################
from pyro.dynamic  import pendulum
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
from pyro.analysis import graphical
##############################################################################


def case( m , g , l , t_max_star , q_star , case_name = 'test ', rax = None , rax2 = None, res = 'reg', legend = 0):
    
    # Additionnal fixed domain dimentionless parameters
    theta_star  = 2.0 * np.pi
    dtheta_star = 1.0 * np.pi
    time_star   = 2.0 * np.pi * 20.0
    
    # Combined system parameters
    omega = np.sqrt( ( g / l  ) )
    mgl   = m * g * l
    
    # Dimentional parameters
    t_max  = t_max_star * mgl
    q      = q_star * mgl
    theta  = theta_star
    dtheta = dtheta_star * omega
    time   = time_star / omega
    J_max  = mgl**2 / omega * time_star * ( ( q_star * theta_star )**2 + t_max_star**2 )
    
    print('\n\nCase :' + case_name )
    print('----------------------------------------------------')
    print(' m=',m,' g=',g,' l=',l,' t_max=', t_max, ' q=', q)
    
    ################################
    # Dynamic system definition
    ################################
    
    sys  = pendulum.InvertedPendulum()

    # kinematic
    sys.lc1 = l

    sys.l1       = sys.lc1
    sys.l_domain = sys.lc1 * 2

    # dynamic
    sys.m1       = m
    sys.I1       = 0
    sys.gravity  = g
    sys.d1       = 0

    sys.u_ub[0]  = + t_max
    sys.u_lb[0]  = - t_max

    sys.x_ub = np.array([ + theta , + dtheta ])
    sys.x_lb = np.array([ - theta , - dtheta ])
    
    ################################
    # Discritized grid
    ################################
    
    if res == 'low' :

        dt = 0.1
        nx = 101
        nu = 11
    
    elif res == 'plus' :
        
        dt = 0.05
        nx = 301
        nu = 101
        
    elif res == 'hi' :
        
        dt = 0.025
        nx = 501
        nu = 101
        
    elif res == 'test' :
        
        dt = 0.5
        nx = 21
        nu = 3
        
    else:
        
        dt = 0.05
        nx = 301
        nu = 21
            
    grid_sys = discretizer.GridDynamicSystem( sys , [nx,nx] , [nu] , dt , True )
    #grid_sys.save_lookup_tables('301_21')

    #grid_sys = discretizer.GridDynamicSystem( sys , [301,301] , [21] , dt , False )
    #grid_sys.load_lookup_tables('301_21')

    ################################
    # Cost function
    ################################

    qcf = costfunction.QuadraticCostFunction.from_sys(sys)

    qcf.xbar = np.array([ 0 , 0 ]) # target
    qcf.INF  = J_max


    qcf.Q[0,0] = q ** 2
    qcf.Q[1,1] = 0.0

    qcf.R[0,0] = 1.0

    qcf.S[0,0] = 0.0
    qcf.S[1,1] = 0.0


    ################################
    # Cost function
    ################################
    
    dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf )


    steps = int( time / dt) 

    dp.compute_steps( steps )
    #dp.solve_bellman_equation( tol = J_min )


    grid_sys.fontsize = 10
    qcf.INF  = 0.1 * J_max
    dp.clean_infeasible_set()
    
    #dp.plot_cost2go()
    #dp.plot_policy()
    #dp.cost2go_fig[0].savefig( case_name + '_cost2go.pdf')
    #dp.policy_fig[0].savefig( case_name + '_policy.pdf')
    
    ##################################
    # Dimensional policy plot
    ##################################

    fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
    fig.canvas.manager.set_window_title( 'dimentionless policy' )
    ax  = fig.add_subplot(1, 1, 1)

    xname = r'$\theta  \; [rad]$'#self.sys.state_label[x] #+ ' ' + self.sys.state_units[x]
    yname = r'$\dot{\theta} \; [rad/sec]$'#self.sys.state_label[y] #+ ' ' + self.sys.state_units[y]
    zname = r'$\tau \; [Nm]$'
    sys.state_label[0] = r'$\theta$'
    sys.state_label[1] = r'$\dot{\theta}$'
    sys.input_label[0] = r'$\tau$'
    
    xrange = 2.0 * np.pi
    yrange = np.pi * np.sqrt( 10 / 1. )
    zrange = 20.

    ax.set_ylabel(yname, fontsize=10)
    ax.set_xlabel(xname, fontsize=10)

    x_level = grid_sys.x_level[ 0 ] 
    y_level = grid_sys.x_level[ 1 ] 


    u = grid_sys.get_input_from_policy( dp.pi , 0 )

    u2 =  u 

    J_grid_nd = grid_sys.get_grid_from_array( u2 ) 

    J_grid_2d = grid_sys.get_2D_slice_of_grid( J_grid_nd , 0 , 1 )

    mesh = ax.pcolormesh( x_level, y_level, J_grid_2d.T, 
                   shading='gouraud' , cmap = 'bwr', vmin = -zrange, vmax = zrange ) #, norm = colors.LogNorm()

    ax.tick_params( labelsize = 10 )
    ax.grid(True)
    ax.set_ylim( -yrange, +yrange)
    ax.set_xlim( -xrange, xrange)

    cbar = fig.colorbar( mesh )

    cbar.set_label(zname, fontsize=10 , rotation=90)

    fig.tight_layout()
    fig.show()
    fig.savefig( case_name + '_policy.pdf')
    fig.savefig( case_name + '_policy.png')
    fig.savefig( case_name + '_policy.jpg')
    

    ##################################
    # Trajectory plot
    ##################################
    
    ctl = dp.get_lookup_table_controller()

    # Simulation
    cl_sys = ctl + sys
    cl_sys.x0   = np.array([-3.14, 0.])
    cl_sys.compute_trajectory( 10 , 6001, 'euler')
    #cl_sys.plot_trajectory('xu')
    #cl_sys.plot_phase_plane_trajectory()
    #cl_sys.animate_simulation()

    tp = graphical.TrajectoryPlotter( sys )
    tp.fontsize = 10
    tp.plot( cl_sys.traj , 'xu')
    #tp.plots[0].set_ylim([-xrange, xrange])
    #tp.plots[1].set_ylim([-yrange, yrange])
    tp.plots[1].set_ylim([-5.5, 5.5])
    tp.plots[2].set_ylim([-zrange, zrange])
    tp.fig.savefig( case_name + '_traj.pdf')
    tp.fig.savefig( case_name + '_traj.png')
    tp.fig.savefig( case_name + '_traj.jpg')
    

    ##################################
    # Dimensionless policy plot
    ##################################

    fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
    fig.canvas.manager.set_window_title( 'dimentionless policy' )
    ax  = fig.add_subplot(1, 1, 1)

    xname = r'$\theta^*$'#self.sys.state_label[x] #+ ' ' + self.sys.state_units[x]
    yname = r'$\dot{\theta}^* = \frac{\dot{\theta}}{\omega}$'#self.sys.state_label[y] #+ ' ' + self.sys.state_units[y]
    zname = r'$\tau^*=\frac{\tau}{mgl}$'

    ax.set_ylabel(yname, fontsize=10)
    ax.set_xlabel(xname, fontsize=10)

    x_level = grid_sys.x_level[ 0 ] * 1
    y_level = grid_sys.x_level[ 1 ] * (1 / omega)


    u = grid_sys.get_input_from_policy( dp.pi , 0 )

    u2 =  u * (1/mgl)

    J_grid_nd = grid_sys.get_grid_from_array( u2 ) 

    J_grid_2d = grid_sys.get_2D_slice_of_grid( J_grid_nd , 0 , 1 )

    mesh = ax.pcolormesh( x_level, y_level, J_grid_2d.T, 
                   shading='gouraud' , cmap = 'bwr') #, norm = colors.LogNorm()

    #mesh.set_clim(vmin=jmin, vmax=jmax)


    ax.tick_params( labelsize = 10 )
    ax.grid(True)

    cbar = fig.colorbar( mesh )

    cbar.set_label(zname, fontsize=10 , rotation=90)

    fig.tight_layout()
    fig.show()
    fig.savefig( case_name + '_dimpolicy.pdf')
    fig.savefig( case_name + '_dimpolicy.png')
    fig.savefig( case_name + '_dimpolicy.jpg')
    
    
    if rax is not None:
    
    ###############################
    # 2D policy regime figure (dtheta = 0 )
    ###############################
    
        n = 101
        x_min = - theta_star - 0.1
        x_max = + theta_star + 0.1
    
        x = np.linspace( x_min, x_max, n) 
        u = np.zeros(n)
    
        for i in range(n):
            ri = 0
            xi = np.array([ x[i] , 0.0 ])
            ti = 0
            u[i] = ctl.c( xi, ri, ti) * (1/mgl)
        
        
        if legend == 1:
            rax.plot( x , u , label= r'$\tau_{max}^* =$ %0.1f' % t_max_star )
        elif legend == 2:
            rax.plot( x , u , label= r'$q^* =$ %0.2f' % q_star )
        else:
            rax.plot( x , u )
            
        rax.set_xlim([ x_min, x_max ])
        rax.set_xlabel( xname, fontsize=10 )
        rax.grid(True)
        rax.tick_params( labelsize = 10 )
        rax.set_ylabel( zname, fontsize=10)
        
    if rax2 is not None:
    
    ###############################
    # 2D policy regime figure (theta = -np.pi )
    ###############################
    
        n = 101
        x_min = - dtheta_star - 0.1
        x_max = + dtheta_star + 0.1
    
        x = np.linspace( x_min, x_max, n) 
        u = np.zeros(n)
    
        for i in range(n):
            ri = 0
            xi = np.array([ -np.pi , x[i] ])
            ti = 0
            u[i] = ctl.c( xi, ri, ti) * (1/mgl)
            
    
        if legend == 1:
            rax2.plot( x , u , label= r'$\tau_{max}^* =$ %0.1f' % t_max_star )
        elif legend == 2:
            rax2.plot( x , u , label= r'$q^* =$ %0.2f' % q_star )
        else:
            rax2.plot( x , u )
            
        rax2.set_xlim([ x_min, x_max ])
        rax2.set_xlabel( yname, fontsize=10 )
        rax2.grid(True)
        rax2.tick_params( labelsize = 10 )
        rax2.set_ylabel( zname, fontsize=10)
    
    
    

def sensitivity( ts , qs , res = 'mid' , name = 'sensitivity' , legend = 1):

    rfig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
    rax  = rfig.add_subplot(1, 1, 1)
    
    rfig2 = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
    rax2  = rfig2.add_subplot(1, 1, 1)
    
    n = ts.size
    
    for i in range(n):
        
        case( m=1 , g=10 , l=1 , t_max_star= ts[i] , q_star= qs[i] , case_name = name + '_level_' + str(i+1) , rax = rax, rax2 = rax2, res = res, legend = legend)
    
    
    # case( m=1 , g=10 , l=1 , t_max_star=0.1 , q_star= 0.05 , case_name = 't1', rax = rax, rax2 = rax2, res = res, legend = 1)
    # case( m=1 , g=10 , l=1 , t_max_star=0.2 , q_star= 0.05 , case_name = 't2', rax = rax, rax2 = rax2, res = res, legend = 1)
    # case( m=1 , g=10 , l=1 , t_max_star=0.3 , q_star= 0.05 , case_name = 't3', rax = rax, rax2 = rax2, res = res, legend = 1)
    # case( m=1 , g=10 , l=1 , t_max_star=0.4 , q_star= 0.05 , case_name = 't4', rax = rax, rax2 = rax2, res = res, legend = 1)
    # case( m=1 , g=10 , l=1 , t_max_star=0.5 , q_star= 0.05 , case_name = 't5', rax = rax, rax2 = rax2, res = res, legend = 1)
    # case( m=1 , g=10 , l=1 , t_max_star=0.6 , q_star= 0.05 , case_name = 't6', rax = rax, rax2 = rax2, res = res, legend = 1)
    # case( m=1 , g=10 , l=1 , t_max_star=0.7 , q_star= 0.05 , case_name = 't7', rax = rax, rax2 = rax2, res = res, legend = 1)
    # case( m=1 , g=10 , l=1 , t_max_star=0.8 , q_star= 0.05 , case_name = 't8', rax = rax, rax2 = rax2, res = res, legend = 1)
    # case( m=1 , g=10 , l=1 , t_max_star=0.9 , q_star= 0.05 , case_name = 't9', rax = rax, rax2 = rax2, res = res, legend = 1)
    
    
    rax.legend( loc = 'upper right' )
    rfig.tight_layout()
    rfig.show()
    rfig.savefig( name + '.pdf')
    rfig.savefig( name + '.png')
    rfig.savefig( name + '.jpg')
    
    rax2.legend( loc = 'upper right' )
    rfig2.tight_layout()
    rfig2.show()
    rfig2.savefig( name + '2.pdf')
    rfig2.savefig( name + '2.png')
    rfig2.savefig( name + '2.jpg')
    
    return (rfig, rax, rfig2, rax2)



####################################
### Main
####################################

res = 'test'

# case( m=1 , g=10 , l=1 , t_max_star=0.5 , q_star= 0.1 , case_name = 'c1', res = res)
# case( m=1 , g=10 , l=2 , t_max_star=0.5 , q_star= 0.1 , case_name = 'c2', res = res)
# case( m=2 , g=10 , l=1 , t_max_star=0.5 , q_star= 0.1 , case_name = 'c3', res = res)
# case( m=1 , g=10 , l=1 , t_max_star=1.0 , q_star= 0.05 , case_name = 'c4', res = res)
# case( m=1 , g=10 , l=2 , t_max_star=1.0 , q_star= 0.05 , case_name = 'c5', res = res)
# case( m=2 , g=10 , l=1 , t_max_star=1.0 , q_star= 0.05 , case_name = 'c6', res = res)
# case( m=1 , g=10 , l=1 , t_max_star=1.0 , q_star= 10.0 , case_name = 'c7', res = res)
# case( m=1 , g=10 , l=2 , t_max_star=1.0 , q_star= 10.0 , case_name = 'c8', res = res)
# case( m=2 , g=10 , l=1 , t_max_star=1.0 , q_star= 10.0 , case_name = 'c9', res = res)

ts = np.array([  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])
qs = np.array([  0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

fig, ax, fig2, ax2 = sensitivity(ts, qs , res = res , name = 'sensitivity_q_cts', legend = 1)

ts = np.array([  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5])
qs = np.array([  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])

fig, ax, fig2, ax2 = sensitivity(ts, qs , res = res , name = 'sensitivity_tau_cts', legend = 2)
    
    