#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Created on Wed Oct 12 20:13:17 2022@author: alex"""import numpy as npimport matplotlib.pyplot as pltfrom scipy.interpolate import RectBivariateSpline as interpol2Dfrom scipy.interpolate import RegularGridInterpolator as rgifrom pyro.control  import controllerfrom pyro.planning import valueiteration################################################################################## RL algo###############################################################################class Q_Learning_2D( valueiteration.ValueIteration_2D ):    """ State is a 2D space grid """        ############################    def __init__(self, grid_sys , cost_function ):                # Dynamic system        self.grid_sys  = grid_sys         # Discretized Dynamic system class        self.sys       = grid_sys.sys     # Base Dynamic system class                # Cost function        self.cf  = cost_function                self.dt = self.grid_sys.dt                # Q Learning Parameters        self.alpha = 0.9   # Discount factor        self.eta   = 0.8   # Learning rate                    ##############################    def initialize(self):        """ initialize cost-to-go and policy """                J_dim = self.grid_sys.xgriddim        Q_dim = ( J_dim[0], J_dim[1] , self.grid_sys.actions_n )                self.Q             = np.zeros( Q_dim , dtype = float )        self.J             = np.zeros( J_dim , dtype = float )        self.action_policy = np.zeros( J_dim , dtype = int   )        self.Jnew          = self.J.copy()        self.Jplot         = self.J.copy()                # Set initial value              for node in range( self.grid_sys.nodes_n ):                              x = self.grid_sys.nodes_state[ node , : ]                                i = self.grid_sys.nodes_index[ node , 0 ]                j = self.grid_sys.nodes_index[ node , 1 ]                                # Final Cost                final_cost = self.cf.h( x )                                # Initial J value                self.J[i,j] = final_cost                                # Initial Q-values                for action in range( self.grid_sys.actions_n ):                                        k = self.grid_sys.actions_index[action,0]                                        self.Q[i,j,k] = final_cost                                             ##############################    def compute_J_from_Q(self):        """ update the J table from Q values """                for node in range( self.grid_sys.nodes_n ):                              i = self.grid_sys.nodes_index[ node , 0 ]                j = self.grid_sys.nodes_index[ node , 1 ]                                self.J[i,j] = self.Q[i,j,:].min()                                    ###############################################    def Q_update_from_index(self, s , a , s_next ):        """         s1: index of initial node        k:  index of taken action        s2: index of resulting node        """                # Get value of initial state        x = self.grid_sys.nodes_state[ s , : ]        i = self.grid_sys.nodes_index[ s , 0 ]        j = self.grid_sys.nodes_index[ s , 1 ]                # Get value of taken action        u = self.grid_sys.actions_input[ a , : ]        k = self.grid_sys.actions_index[ a , 0 ]                # Get value of next state        x_next = self.grid_sys.nodes_state[ s_next , : ]        i_next = self.grid_sys.nodes_index[ s_next , 0 ]        j_next = self.grid_sys.nodes_index[ s_next , 1 ]                # Compute optimal cost-to-go of next state minimize over all u_next actions        self.J[i_next,j_next] = self.Q[i_next,j_next,:].min()                        if self.sys.isavalidstate(x_next):                        J_next = self.J[i_next,j_next]                        # No need to interpolate since this function is based on x_next index                        # Get interpolation of current cost space            #J_interpol = interpol2D( self.grid_sys.xd[0] , self.grid_sys.xd[1] , self.J , bbox=[None, None, None, None], kx=1, ky=1,)            #J_next = J_interpol( x_next[0] , x_next[1] )                        y = self.sys.h(x, u, 0)                        Q_sample = self.cf.g(x, u, y, 0) * self.dt + self.alpha * J_next                    else:                        Q_sample = self.cf.INF                            # Q update        error          = Q_sample      - self.Q[i,j,k]        self.Q[i,j,k]  = self.Q[i,j,k] + self.eta * error                # Action policy update        self.action_policy[i,j] = self.Q[i,j,:].argmin()                    ###############################################    def Q_update_from_nearest_values(self, x , u , x_next ):        """         Find closest nodes on the grid and do a Q update        """                # Find closest index of x        i = (np.abs(self.grid_sys.xd[0]-x[0])).argmin()        j = (np.abs(self.grid_sys.xd[1]-x[1])).argmin()                s = self.grid_sys.x_grid2node[i,j]                # Find closest index of u        k = (np.abs(self.grid_sys.ud[0]-u[0])).argmin()                # Find closest index        i_next = (np.abs(self.grid_sys.xd[0]-x_next[0])).argmin()        j_next = (np.abs(self.grid_sys.xd[1]-x_next[1])).argmin()                s_next = self.grid_sys.x_grid2node[i_next,j_next]                self.Q_update_from_index( s , k , s_next )                                                                