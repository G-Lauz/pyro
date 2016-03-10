# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:09:12 2016

@author: alex
"""

from AlexRobotics.control import linear        as RCL

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

'''
################################################################################
'''


class RRT:
    """ Rapid Random Trees search algorithm """
    
    ############################
    def __init__(self, sys , x_start ):
        
        self.DS = sys          # Dynamic system class
        
        self.x_start = x_start  # origin of the graph
        
        self.start_node = Node( self.x_start , sys.ubar  , None )
        
        self.Nodes = []
        self.Nodes.append( self.start_node )
        
        self.solution = None
        
        # Params
        self.dt  = 0.05
        self.INF = 10000
        self.eps = 0.2
        
        self.goal_radius          = 2        
        self.alpha                = 0.9    # prob of random exploration
        self.max_nodes            = 25000  # maximum number of nodes
        self.max_distance_compute = 500    # maximum number of nodes to check distance
        
        self.traj_ctl_kp          = 25
        self.traj_ctl_kd          = 2
        
        self.discretizeactions()
        
        
        #############################
    def discretizeactions(self, Nu0 = 3 ):
        
        self.U = np.linspace( self.DS.u_lb[0]  , self.DS.u_ub[0]  , Nu0  )
        
        
    ############################
    def rand_state(self):    
        """ Sample a random state """
        
        ranges = self.DS.x_ub - self.DS.x_lb
        
        x_random = np.random.rand( self.DS.n ) * ranges + self.DS.x_lb
        
        return x_random
        
        
    ############################
    def nearest_neighbor(self, x_target ):    
        """ Sample a random state """
        
        closest_node = None
        min_distance = self.INF
        
        if len(self.Nodes) < self.max_distance_compute + 1 :
            # Brute force        
            for node in self.Nodes:
                d = node.distanceTo( x_target )
                if d < min_distance:
                    min_distance = d
                    closest_node = node
        
        else:
            # Check only last X nodes
            for i in xrange(self.max_distance_compute):
                node = self.Nodes[ len(self.Nodes) - i - 1 ]
                d = node.distanceTo( x_target )
                if d < min_distance:
                    min_distance = d
                    closest_node = node
            
                
        return closest_node
        
        
    ############################
    def select_control_input(self, x_target , closest_node ):    
        """ Sample a random state """
        
        new_node     = None
        min_distance = self.INF
        
        for u in self.U:
            if self.DS.m == 1:
                u = np.array([u])
            x_next = closest_node.x + self.DS.fc( closest_node.x , u ) * self.dt
            node   = Node( x_next , u , closest_node )
            d = node.distanceTo( x_target )
            if d < min_distance:
                min_distance = d
                new_node     = node
                
        return new_node
        
    
    ############################
    def one_step(self):    
        """ """
        x_random  = self.rand_state()
        node_near = self.nearest_neighbor( x_random )
        new_node  = self.select_control_input( x_random , node_near )
        
        self.Nodes.append( new_node )
        
        
    ############################
    def compute_steps(self , n , plot = False ):    
        """ """
        for i in xrange( n ):
            self.one_step()
            
        if plot:
            self.plot_2D_Tree()
    
           
        
    ############################
    def find_path_to_goal(self, x_goal ):
        """ """
        
        self.x_goal = x_goal
        
        succes   = False
        
        no_nodes = 0
        
        while not succes:
            
            # Exploration:
            if np.random.rand() > self.alpha :
                x_random = x_goal
            else:
                x_random  = self.rand_state()
                
            # Expansion
            node_near = self.nearest_neighbor( x_random )
            new_node  = self.select_control_input( x_random , node_near )
            
            self.Nodes.append( new_node )
            
            # Distance to goal
            d = new_node.distanceTo( x_goal )
            
            # Print
            no_nodes = no_nodes + 1
            #print '\nNumber of Nodes = ', no_nodes
            
            # Succes?
            if d < self.goal_radius:
                succes = True
                self.goal_node = new_node
                
            # Tree reset
            if no_nodes == self.max_nodes:
                print '\nSearch Fail: Reseting Tree'
                #self.plot_2D_Tree()
                no_nodes = 0
                self.Nodes = []
                self.Nodes.append( self.start_node )
                
        
        print '\nSucces!!!!: Path to goal found'
        
        
        # Compute Path
        self.compute_path_to_goal()
        
                
    ############################
    def compute_path_to_goal(self):
        """ """
        
        node = self.goal_node
        
        t      = 0
        
        x_list = []
        u_list = []
        t_list = []
        
        self.path_node_list = []
        self.path_node_list.append( node )
        
        while node.distanceTo( self.x_start ) > self.eps:
            
            self.path_node_list.append( node )
            
            x_list.append( node.x )
            u_list.append( node.u )
            t_list.append( t )
            
            t = t - self.dt
            
            # Previous Node
            node  = node.P 
        
        # Arrange Time array
        t = np.array( t_list )
        t = t - t.min()
        t = np.flipud( t )
        self.time_to_goal = t.max()
        
        # Arrange Input array
        u = np.array( u_list ).T
        u = np.fliplr( u )
        
        # Arrange State array
        x = np.array( x_list ).T
        x = np.fliplr( x )
            
        self.solution = [ x , u , t ]
        
        
    ############################
    def open_loop_controller(self, x , t ):
        """ feedback law """
        
        if self.solution == None:
            
            u = self.DS.ubar
            
            return u
        
        else:
            
            # Find time index
            times = self.solution[2]
            i = (np.abs(times - t)).argmin()
            
            # Find associated control input
            inputs = self.solution[1][0]
            u      = np.array( [ inputs[i] ] )
            
            return u
            
    ############################
    def trajectory_controller(self, x , t ):
        """ feedback law """
        
        if self.solution == None:
            
            u = self.DS.ubar
            
            return u
        
        else:
            
            # Find time index
            times = self.solution[2]
            i = (np.abs(times - t)).argmin()
            
            # Find associated control input
            inputs = self.solution[1][0]
            u_bar  = np.array( [ inputs[i] ] )
            
            # Find associated state and compute error
            states   = self.solution[0]
            x_target = states[:,i]
            
            # No action pass trajectory time
            if t > self.time_to_goal:
                u_bar = self.DS.ubar
                x_target = self.x_goal
            
            error    = x_target - x
            
            # Error feedback
            if self.DS.n == 2:
                """ 1 DOF manipulator """
                K     = np.array([ self.traj_ctl_kp , self.traj_ctl_kd ])
                u_fdb = np.dot( K , error )
                
            elif self.DS.n == 4:
                """ 2 DOF manipulator """
                u1     = np.dot( np.array([ self.traj_ctl_kp , 0  , self.traj_ctl_kd ,  0 ]) , error ) 
                u2     = np.dot( np.array([ 0  , self.traj_ctl_kp ,  0 , self.traj_ctl_kd ]) , error ) 
                u_fdb  = np.array([ u1 , u2 ])
                
            else:
                u_fdb = 0
            
            return u_bar + u_fdb
            
                
                
    ############################
    def plot_2D_Tree(self):
        """ """
        
        self.y1axis = 0  # State to plot on y1 axis
        self.y2axis = 1  # State to plot on y2 axis
        
        self.y1min = self.DS.x_lb[ self.y1axis ]
        self.y1max = self.DS.x_ub[ self.y1axis ]
        self.y2min = self.DS.x_lb[ self.y2axis ]
        self.y2max = self.DS.x_ub[ self.y2axis ]
        
        self.phasefig = plt.figure(figsize=(3, 2),dpi=300, frameon=True)
        self.ax       = self.phasefig.add_subplot(111)
        
        for node in self.Nodes:
            if not(node.P==None):
                line = self.ax.plot( [node.x[0],node.P.x[0]] , [node.x[1],node.P.x[1]] , 'o-')
                
        if not self.solution == None:
            for node in self.path_node_list:
                if not(node.P==None):
                    line = self.ax.plot( [node.x[0],node.P.x[0]] , [node.x[1],node.P.x[1]] , 'r')
        
        
        plt.xlabel(self.DS.state_label[ self.y1axis ] + ' ' + self.DS.state_units[ self.y1axis ] , fontsize=6)
        plt.ylabel(self.DS.state_label[ self.y2axis ] + ' ' + self.DS.state_units[ self.y2axis ] , fontsize=6)
        plt.xlim([ self.y1min , self.y1max ])
        plt.ylim([ self.y2min , self.y2max ])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
                        
    
        
        
class Node:
    """ node of the graph """
    
    ############################
    def __init__(self, x , u , parent ):
        
        self.x = x  # Node coordinates in the state space
        self.u = u  # Control inputs used to get there
        self.P = parent # Previous node
        
    
    ############################
    def distanceTo(self, x_other ):
        """ Compute distance to otherNode """
        
        return np.linalg.norm( self.x - x_other )
        
        
        
        
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    pass
        
        
