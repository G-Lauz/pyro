# Pyro 

An object-based toolbox for robot dynamic simulation, analysis, control and planning. 

## A library of robots and dynamic systems models
<table>
  <tr>
    <th>
    <img src="https://user-images.githubusercontent.com/16725496/162986261-b3f6950b-e417-403b-8e81-81b30a542d6c.gif" alt="rocket" width="400"/>
    </th>
    <th>
    <img src="https://user-images.githubusercontent.com/16725496/163005905-ad2205b0-150d-44de-bd43-a3b31a0bf10e.gif" alt="cartpole" width="400"/>
    </th> 
  </tr>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/16725496/163005883-5ec9b6f8-d8ab-44b1-bc9d-ac5ca2d6b4a9.gif" alt="drone" width="400"/>
    </td>
    <td>
    <img src="https://user-images.githubusercontent.com/16725496/163005950-665132ae-c1d5-486c-8bf1-3c3fa9aa4140.gif" alt="mass-spring" width="400"/>
    </td> 
  </tr>
</table>

## A collection of controller synthesis tools:
<table>
  <tr>
    <th>
    <img src="https://www.alexandregirard.com/IMG/cost2go_animation.gif" alt="cost2go" width="400"/>
    </th>
    <th>
    <img src="https://www.alexandregirard.com/IMG/policy_animation.gif" alt="cost2go" width="400"/>
    </th> 
  </tr>
</table>



## Library Architecture ##

The concept of this toolbox is a hierachy of "dynamic system" objects, from the most generic representation (any non-linear differential equations) to more system specific representations such as mechanical system (second order equations), linear state space, manipulator equations, etc. This structure is then leveraged by analysis tools, from generic tools that work for all sub-class of dynamic systems such as running simulation and phase-plane analysis, to system-specific tools that leverage specific system propreties such as modal analysis for linear sub-class:

<img width="900" src="https://user-images.githubusercontent.com/16725496/163312294-e33d791f-9cc0-48e1-acb3-8a0ebfc0c067.jpg" class="center">

The core of the library is a mother "dyanmic system" class defined by a differential equation $\dot{x} = f(x,u,t)$, an output equation $y = h(x,u,t)$ and a foward kinematic equation $lines = fwd_kinematic(x,u,t)$ that is used for generating animations:

<img width="900" src="https://user-images.githubusercontent.com/16725496/163312300-faa7fe2c-178e-4c58-ae6c-4b256fd9ab92.jpg" class="center">

By creating a class defining these three base functions, most of the library tools can then be used directly to analyze or to generate model-based controllers.

## How to use ##

To learn how to use pyro, see the following notebook tutorials hosted on colab:

1.   [The Dynamic System class and basic functionnality](https://colab.research.google.com/drive/18eEL-n-dv9JZz732nFCMtqMThDcfD2Pr?usp=sharing)
2.   [Creating a custom dynamic class](https://colab.research.google.com/drive/1ILfRpL1zgiQZBOxwtbbpe0nl2znvzdWl?usp=sharing)
3.   [Closed-loop system and controllers objects](https://colab.research.google.com/drive/1mog1HAFN2NFEdw6sPudzW2OaTk_li0Vx?usp=sharing)
4.   The Linear System class (comin soon..)
4.   The Mechanical System class (coming soon..)
5.   [The Manipulator Robot class](https://colab.research.google.com/drive/1OILAhXRxM1r5PEB1BWaYtbR147Ff3gr1?usp=sharing)

Also see exemples scripts in pyro/examples/ 


## Installation ##

### Dependencies ####
Pyro is built only using core python librairies: 
* numpy
* scipy
* matplotlib

### Using in Colab ###

```
!git clone https://github.com/SherbyRobotics/pyro
import sys
sys.path.append('/content/pyro')
import pyro
```

### Using with Anaconda and Spyder IDE ###
1. Download anaconda distribution (including spyder IDE) available here: https://www.anaconda.com/products/individual

2. Dowload pyro source code. 
A simple option for development is simply to clone the repo:
```bash
git clone https://github.com/SherbyRobotics/pyro.git
```
then add the pyro folder to the pythonpath variable of your environment. In spyder this option is found in the menu at python/PYTHONPATH manager.

3. Change the graphical backend in Spyder for enabling animation
If graphical animations are not working, try changing the graphics backend in the menu at python/Preferences/IPython console/Backend. Inline does not allow animations, it is best to use Automatic (for Windows and Ubuntu) or OS X (for Mac).



## Pyro internal structure ##

### Dynamic objects ###

At the core of pyro is a mother-class representing generic non-linear dynamic systems, with the following nomemclature:

<img width="500" src="https://user-images.githubusercontent.com/16725496/116826021-fd9b7a80-ab5f-11eb-8e50-d7361094cbee.png" class="center">

The main hierachy of sub-class of dynamic systems and implemented exemples are:
- Linear System
  - Transfer function 
  - Exemples: mass-spring-damper
- Mechanical System
  - Manipulator Robot
    - Exemples: two link plananr robot
    - Exemples: five link plannar robot
    - Exemples: three link robot
  - Exemples: single pendulum
  - Exemples: double pendulum
  - Exemples: cart-pole
  - Exemples: planar drone
  - Exemples: rocket
- Exemples: bicycle model (planar vehicle)

### Analysis tool ###

- Copmuting simulation
- Phase-plane analysis
- Graphical animated output of the simulations
- Cost function computation
- Linearisation (from any dynamic class to the state-space class)
- Modal analysis
- Pole/zero computation
- Bode plot


### Controller objects ###

Controller objects can be used to closed the loop with an operation generating a closed-loop dynamic system:

closed-loop system = controller + open-loop system

For "memoryless" controller, this operation is

<img width="500" src="https://user-images.githubusercontent.com/16725496/116826519-59ff9980-ab62-11eb-8256-6a9f4a3f4f0f.png" class="center">

Available control algorithms: 
- PID
- LQR
- Computed-Torque
- Sliding-mode controller
- End-point Impedance
- Value-Iteration
- End-point impedance controller for robot arms
- End-point trajectory controller for robot arms


### Planner objects ###

Available planner algorithm:
1. RRT tree search
2. Direct collocation trajectory optimisation
3. Value-iteration 








