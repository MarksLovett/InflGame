import numpy as np
import torch
import matplotlib.tri as tri

from influencer_games.utils.utilities import *

def simplex_setup(refinement:int = 4):
        # Sets up the simplex (points, corner, grid)
        # OUTPUTS: 
        #   mesh:for the grid on the simplex
        #   corners: the corners of the simplex

        r0=np.array([0,0])
        r1=np.array([1,0])
        r2=np.array([1/2.,np.sqrt(3)/2.])
        corners =np.array([r0,r1,r2])
        triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
        refiner = tri.UniformTriRefiner(triangle)
        trimesh = refiner.refine_triangulation(subdiv=refinement)
        trimesh_fine = refiner.refine_triangulation(subdiv=refinement)
        return r2,corners,triangle,trimesh

def simplex_bin_setup(domain_bounds):
        corners=domain_bounds[1]
        trimesh=domain_bounds[3]
        bin_points_xy=np.array([[x,y]  for x,y in zip(trimesh.x, trimesh.y)])
        bin_points=np.array([xy2ba(x,y,corners)  for x,y in zip(trimesh.x, trimesh.y)])
        for bin_point_id in range(len(bin_points)):
                bin_point=projection_onto_siplex(torch.tensor(bin_points[bin_point_id])).numpy()[0]
                bin_point=np.round(bin_point,decimals=5)
                if any(x<=0 for x in bin_point):
                        if any(x==1 for x in bin_point):
                                i=np.where(bin_point==1)[0][0]
                                bin_point[i]-=.001
                                bin_point[i-1]+=.0005
                                if i==2:
                                        bin_point[i-2]+=.0005
                                else:
                                        bin_point[i+1]+=.0005
                                bin_points[bin_point_id]=bin_point
                        elif any(x==0 for x in bin_point):
                                i=np.where(bin_point==0)[0][0]
                                bin_point[i]+=.001
                                bin_point[i-1]-=.0005
                                if i==2:
                                        bin_point[i-2]-=.0005
                                else:
                                        bin_point[i+1]-=.0005
                                        
                bin_points[bin_point_id]=bin_point
        return bin_points,bin_points_xy

def xy2ba(x:torch.Tensor,
          y:torch.Tensor,
          corners:np.ndarray
              )->torch.Tensor:
        # Coverts the Cartesian coordinates to the Baryocentric coordinates
        # OUTPUTS: 
        #   mesh:for the grid on the simplex
        #   corners: the corners of the simplex

        corner_x=corners.T[0]
        corner_y=corners.T[1]
        x_1=corner_x[0]
        x_2=corner_x[1]
        x_3=corner_x[2]
        y_1=corner_y[0]
        y_2=corner_y[1]
        y_3=corner_y[2]
        l1=((y_2-y_3)*(x-x_3)+(x_3-x_2)*(y-y_3))/((y_2-y_3)*(x_1-x_3)+(x_3-x_2)*(y_1-y_3))
        l2=((y_3-y_1)*(x-x_3)+(x_1-x_3)*(y-y_3))/((y_2-y_3)*(x_1-x_3)+(x_3-x_2)*(y_1-y_3))
        l3=1-l1-l2
        return np.array([l1,l2,l3])


def ba2xy(x:torch.Tensor,
          corners:np.ndarray,
          )->torch.Tensor:
        #Coverts the Baryocentric coordinates to the Cartesian coordinates
        #INPUT
        # x: array of 3-dim ba coordinates
        #OUTPUT
        # corners: coordinates of corners of ba coordinate system
        
        return torch.matmul(torch.tensor(corners).T,x.T).T


def projection_onto_siplex(Y:torch.Tensor,
                           )->torch.Tensor:
        # For mult-variate games on the simplex, takes the position vector of a player and maps it to the simplex
        # INPUTS:
        #   Y: is the postion vector of the ith player
        # Outputs: 
        #   X: is the postion vector projected onto the simplex. 
        
        D = Y.shape
        Y=Y.reshape(1,list(D)[0])
        N,D=Y.shape
        X, _ = torch.sort(Y, dim=1, descending=True)
        Xtmp = (torch.cumsum(X, dim=1) - 1) / torch.arange(1, D + 1, dtype=torch.float32)
        Xtmp = Xtmp.repeat(N, 1)
        X = torch.maximum(Y - Xtmp[torch.arange(N).unsqueeze(1), torch.sum(X > Xtmp, dim=1) - 1], torch.tensor(0.0))
        
        return X

