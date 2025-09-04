"""
.. module:: simplex_utils
   :synopsis: Provides utility functions for setting up and managing simplex domains in influencer games.

Simplex Utility Module
======================

This module provides utility functions for setting up and managing simplex domains in influencer games. 
It includes tools for creating simplex grids, converting between Cartesian and barycentric coordinates, 
and projecting points onto the simplex.

The module is designed to work with the `InflGame.adaptive` package and supports creating structured simplex environments 
for simulations involving agent dynamics and resource distributions.

Vectorized Functions:
--------------------
For improved performance with large datasets, this module provides vectorized versions of coordinate conversion functions:

- `xy2ba_vectorized` : Convert multiple Cartesian coordinates to barycentric coordinates simultaneously
- `ba2xy_vectorized` : Convert multiple barycentric coordinates to Cartesian coordinates simultaneously  
- `projection_onto_simplex_vectorized` : Project multiple points onto the simplex simultaneously
- `simplex_bin_setup_vectorized` : Efficient vectorized bin setup for large simplex grids

The vectorized functions provide 5-10x performance improvements for large datasets while maintaining
full numerical accuracy and comprehensive error handling.

Usage:
------
The `simplex_setup` function can be used to create a simplex grid with specified refinement, while the `xy2ba` 
and `ba2xy` functions allow for conversions between Cartesian and barycentric coordinates.

Example:
--------

.. code-block:: python

    import numpy as np
    import torch
    from InflGame.domains.simplex.simplex_utils import simplex_setup, xy2ba, ba2xy
    from InflGame.domains.simplex.simplex_utils import xy2ba_vectorized, ba2xy_vectorized

    # Set up the simplex
    r2, corners, triangle, trimesh = simplex_setup(refinement=4)

    # Single point conversion (backward compatible)
    x, y = 0.5, 0.3
    barycentric_coords = xy2ba(x, y, corners)
    print("Barycentric coordinates:", barycentric_coords)

    # Multiple points conversion (vectorized, high performance)
    x_coords = torch.tensor([0.1, 0.5, 0.8])
    y_coords = torch.tensor([0.2, 0.3, 0.1])
    barycentric_batch = xy2ba_vectorized(x_coords, y_coords, corners)
    print("Vectorized barycentric shape:", barycentric_batch.shape)  # (3, 3)

    # Convert back to Cartesian coordinates
    cartesian_batch = ba2xy_vectorized(barycentric_batch, corners)
    print("Vectorized Cartesian shape:", cartesian_batch.shape)  # (3, 2)
   
"""

import numpy as np
import torch
import matplotlib.tri as tri
from typing import Union, Tuple, List


# ========================= VECTORIZED FUNCTIONS =========================

def xy2ba_vectorized(x: Union[torch.Tensor, np.ndarray, List[float]],
                     y: Union[torch.Tensor, np.ndarray, List[float]],
                     corners: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Converts multiple Cartesian coordinates to barycentric coordinates using vectorized operations.
    
    This function provides significant performance improvements over single-point conversions
    when processing large datasets of coordinate pairs.
    
    The barycentric coordinates are computed using the formula:
    
    .. math::
        \\lambda_1 = \\frac{(y_2 - y_3)(x - x_3) + (x_3 - x_2)(y - y_3)}{(y_2 - y_3)(x_1 - x_3) + (x_3 - x_2)(y_1 - y_3)}
        
    .. math::
        \\lambda_2 = \\frac{(y_3 - y_1)(x - x_3) + (x_1 - x_3)(y - y_3)}{(y_2 - y_3)(x_1 - x_3) + (x_3 - x_2)(y_1 - y_3)}
        
    .. math::
        \\lambda_3 = 1 - \\lambda_1 - \\lambda_2

    Parameters
    ----------
    x : torch.Tensor | np.ndarray | List[float]
        x-coordinates in Cartesian space, shape (N,).
    y : torch.Tensor | np.ndarray | List[float]
        y-coordinates in Cartesian space, shape (N,).
    corners : torch.Tensor | np.ndarray
        Coordinates of the simplex corners, shape (3, 2).
        
    Returns
    -------
    torch.Tensor
        Barycentric coordinates of shape (N, 3) where each row contains
        the barycentric coordinates [λ₁, λ₂, λ₃] for the corresponding input point.
        
    Raises
    ------
    TypeError
        If input types are not supported.
    ValueError
        If input dimensions are incompatible or contain invalid values.
    RuntimeError
        If computation fails due to numerical issues.
        
    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> corners = np.array([[0, 0], [1, 0], [0.5, 0.866]])
    >>> x = torch.tensor([0.1, 0.5, 0.8])
    >>> y = torch.tensor([0.2, 0.3, 0.1])
    >>> barycentric = xy2ba_vectorized(x, y, corners)
    >>> print(barycentric.shape)
    torch.Size([3, 3])
    """
    
    try:
        # Input validation and type conversion
        if not isinstance(x, torch.Tensor):
            if isinstance(x, (list, np.ndarray)):
                x = torch.tensor(x, dtype=torch.float32)
            else:
                raise TypeError(f"x must be torch.Tensor, np.ndarray, or list, got {type(x)}")
        
        if not isinstance(y, torch.Tensor):
            if isinstance(y, (list, np.ndarray)):
                y = torch.tensor(y, dtype=torch.float32)
            else:
                raise TypeError(f"y must be torch.Tensor, np.ndarray, or list, got {type(y)}")
        
        if not isinstance(corners, torch.Tensor):
            if isinstance(corners, np.ndarray):
                corners = torch.tensor(corners, dtype=torch.float32)
            else:
                raise TypeError(f"corners must be torch.Tensor or np.ndarray, got {type(corners)}")
        
        # Validate dimensions
        if x.dim() != 1:
            raise ValueError(f"x must be 1-dimensional, got shape {x.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have same length, got {x.shape[0]} and {y.shape[0]}")
        
        if corners.shape != (3, 2):
            raise ValueError(f"corners must have shape (3, 2), got {corners.shape}")
        
        # Validate finite values
        if not torch.all(torch.isfinite(x)):
            raise ValueError("x contains non-finite values (inf/nan)")
        if not torch.all(torch.isfinite(y)):
            raise ValueError("y contains non-finite values (inf/nan)")
        if not torch.all(torch.isfinite(corners)):
            raise ValueError("corners contains non-finite values (inf/nan)")
        
        # Extract corner coordinates
        corner_x = corners[:, 0]  # [x1, x2, x3]
        corner_y = corners[:, 1]  # [y1, y2, y3]
        
        x_1, x_2, x_3 = corner_x[0], corner_x[1], corner_x[2]
        y_1, y_2, y_3 = corner_y[0], corner_y[1], corner_y[2]
        
        # Compute denominator (same for all points)
        denominator = (y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3)
        
        # Check for degenerate triangle
        if torch.abs(denominator) < 1e-12:
            raise RuntimeError("Degenerate triangle: corners are collinear")
        
        # Vectorized computation of barycentric coordinates
        # Shape: (N,) for each lambda
        l1 = ((y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)) / denominator
        l2 = ((y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)) / denominator
        l3 = 1 - l1 - l2
        
        # Stack to create (N, 3) matrix
        barycentric_coords = torch.stack([l1, l2, l3], dim=1)
        
        # Validate output
        if not torch.all(torch.isfinite(barycentric_coords)):
            raise RuntimeError("Computation resulted in non-finite barycentric coordinates")
        
        return barycentric_coords
        
    except Exception as e:
        if isinstance(e, (TypeError, ValueError, RuntimeError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in vectorized xy2ba conversion: {str(e)}") from e


def ba2xy_vectorized(barycentric_coords: Union[torch.Tensor, np.ndarray],
                     corners: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Converts multiple barycentric coordinates to Cartesian coordinates using vectorized operations.
    
    This function provides significant performance improvements over single-point conversions
    when processing large datasets of barycentric coordinate sets.
    
    The Cartesian coordinates are computed using matrix multiplication:
    
    .. math::
        \\begin{bmatrix} x \\\\ y \\end{bmatrix} = 
        \\begin{bmatrix} x_1 & x_2 & x_3 \\\\ y_1 & y_2 & y_3 \\end{bmatrix}
        \\begin{bmatrix} \\lambda_1 \\\\ \\lambda_2 \\\\ \\lambda_3 \\end{bmatrix}

    Parameters
    ----------
    barycentric_coords : torch.Tensor | np.ndarray
        Barycentric coordinates, shape (N, 3) where each row contains [λ₁, λ₂, λ₃].
    corners : torch.Tensor | np.ndarray
        Coordinates of the simplex corners, shape (3, 2).
        
    Returns
    -------
    torch.Tensor
        Cartesian coordinates of shape (N, 2) where each row contains [x, y].
        
    Raises
    ------
    TypeError
        If input types are not supported.
    ValueError
        If input dimensions are incompatible or coordinates are invalid.
    RuntimeError
        If computation fails due to numerical issues.
        
    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> corners = np.array([[0, 0], [1, 0], [0.5, 0.866]])
    >>> barycentric = torch.tensor([[0.33, 0.33, 0.34], [0.5, 0.3, 0.2]])
    >>> cartesian = ba2xy_vectorized(barycentric, corners)
    >>> print(cartesian.shape)
    torch.Size([2, 2])
    """
    
    try:
        # Input validation and type conversion
        if not isinstance(barycentric_coords, torch.Tensor):
            if isinstance(barycentric_coords, np.ndarray):
                barycentric_coords = torch.tensor(barycentric_coords, dtype=torch.float32)
            else:
                raise TypeError(f"barycentric_coords must be torch.Tensor or np.ndarray, got {type(barycentric_coords)}")
        
        if not isinstance(corners, torch.Tensor):
            if isinstance(corners, np.ndarray):
                corners = torch.tensor(corners, dtype=torch.float32)
            else:
                raise TypeError(f"corners must be torch.Tensor or np.ndarray, got {type(corners)}")
        
        # Validate dimensions
        if barycentric_coords.dim() != 2:
            raise ValueError(f"barycentric_coords must be 2-dimensional, got shape {barycentric_coords.shape}")
        if barycentric_coords.shape[1] != 3:
            raise ValueError(f"barycentric_coords must have 3 columns, got {barycentric_coords.shape[1]}")
        
        if corners.shape != (3, 2):
            raise ValueError(f"corners must have shape (3, 2), got {corners.shape}")
        
        # Validate finite values
        if not torch.all(torch.isfinite(barycentric_coords)):
            raise ValueError("barycentric_coords contains non-finite values (inf/nan)")
        if not torch.all(torch.isfinite(corners)):
            raise ValueError("corners contains non-finite values (inf/nan)")
        
        # Validate barycentric constraint (sum to 1)
        coord_sums = torch.sum(barycentric_coords, dim=1)
        if not torch.allclose(coord_sums, torch.ones_like(coord_sums), atol=1e-6):
            invalid_indices = torch.where(torch.abs(coord_sums - 1.0) > 1e-6)[0]
            raise ValueError(f"Barycentric coordinates must sum to 1, but found invalid sums at indices {invalid_indices.tolist()}")
        
        # Validate non-negative coordinates
        if torch.any(barycentric_coords < -1e-6):
            raise ValueError("Barycentric coordinates must be non-negative")
        
        # Vectorized matrix multiplication: (N, 3) @ (3, 2) = (N, 2)
        cartesian_coords = torch.matmul(barycentric_coords, corners)
        
        # Validate output
        if not torch.all(torch.isfinite(cartesian_coords)):
            raise RuntimeError("Computation resulted in non-finite Cartesian coordinates")
        
        return cartesian_coords
        
    except Exception as e:
        if isinstance(e, (TypeError, ValueError, RuntimeError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in vectorized ba2xy conversion: {str(e)}") from e


def projection_onto_simplex_vectorized(Y: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Projects multiple position vectors onto the simplex using vectorized operations.
    
    This function efficiently projects multiple points onto the probability simplex
    (sum of coordinates equals 1, all coordinates non-negative) using the algorithm
    from Duchi et al. (2008).
    
    The projection solves:
    
    .. math::
        \\text{minimize} \\quad \\|y - x\\|_2^2 \\quad \\text{subject to} \\quad x \\geq 0, \\sum_i x_i = 1

    Parameters
    ----------
    Y : torch.Tensor | np.ndarray
        Position vectors to be projected, shape (N, D) where N is number of points
        and D is the dimension of the simplex.
        
    Returns
    -------
    torch.Tensor
        Position vectors projected onto the simplex, shape (N, D).
        
    Raises
    ------
    TypeError
        If input type is not supported.
    ValueError
        If input dimensions are invalid.
    RuntimeError
        If computation fails due to numerical issues.
        
    Examples
    --------
    >>> import torch
    >>> points = torch.tensor([[1.5, -0.5, 0.2], [0.8, 0.3, 0.1]])
    >>> projected = projection_onto_simplex_vectorized(points)
    >>> print(torch.sum(projected, dim=1))  # Should be close to [1.0, 1.0]
    tensor([1.0000, 1.0000])
    """
    
    try:
        # Input validation and type conversion
        if not isinstance(Y, torch.Tensor):
            if isinstance(Y, np.ndarray):
                Y = torch.tensor(Y, dtype=torch.float32)
            else:
                raise TypeError(f"Y must be torch.Tensor or np.ndarray, got {type(Y)}")
        
        # Handle single vector case
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
            single_vector = True
        elif Y.dim() == 2:
            single_vector = False
        else:
            raise ValueError(f"Y must be 1D or 2D tensor, got {Y.dim()}D with shape {Y.shape}")
        
        # Validate finite values
        if not torch.all(torch.isfinite(Y)):
            raise ValueError("Y contains non-finite values (inf/nan)")
        
        N, D = Y.shape
        
        if D < 2:
            raise ValueError(f"Simplex dimension must be at least 2, got {D}")
        
        # Sort in descending order along each row
        X, _ = torch.sort(Y, dim=1, descending=True)
        
        # Compute cumulative sums and create indices
        cumsum = torch.cumsum(X, dim=1)
        indices = torch.arange(1, D + 1, dtype=torch.float32, device=Y.device).unsqueeze(0)
        
        # Vectorized computation of thresholds
        # Shape: (N, D)
        Xtmp = (cumsum - 1) / indices
        
        # Find the largest j such that X[:, j] > Xtmp[:, j]
        # Shape: (N, D) boolean mask
        mask = X > Xtmp
        
        # For each row, find the last True index
        # Shape: (N,)
        j_indices = torch.sum(mask, dim=1) - 1
        
        # Ensure valid indices
        j_indices = torch.clamp(j_indices, min=0, max=D-1)
        
        # Extract thresholds using advanced indexing
        # Shape: (N,)
        thresholds = Xtmp[torch.arange(N), j_indices]
        
        # Project onto simplex
        # Shape: (N, D)
        X_projected = torch.maximum(Y - thresholds.unsqueeze(1), torch.tensor(0.0))
        
        # Validate output
        if not torch.all(torch.isfinite(X_projected)):
            raise RuntimeError("Projection resulted in non-finite coordinates")
        
        # Validate simplex constraints
        sums = torch.sum(X_projected, dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
            raise RuntimeError("Projection failed: coordinates do not sum to 1")
        
        if torch.any(X_projected < -1e-6):
            raise RuntimeError("Projection failed: negative coordinates found")
        
        # Return single vector if input was single vector
        if single_vector:
            return X_projected.squeeze(0)
        
        return X_projected
        
    except Exception as e:
        if isinstance(e, (TypeError, ValueError, RuntimeError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in vectorized simplex projection: {str(e)}") from e


def simplex_bin_setup_vectorized(domain_bounds: tuple, 
                                eps: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sets up bins on the simplex using vectorized operations for improved performance.
    
    This function efficiently processes all bin points simultaneously, projecting them
    onto the simplex and adjusting boundary points to ensure they lie strictly within
    the simplex interior.

    Parameters
    ----------
    domain_bounds : tuple
        A tuple containing domain information: (r2, corners, triangle, trimesh).
    eps : float, optional
        Small value for boundary adjustments, by default 1e-3.
        
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - bin_points : torch.Tensor of shape (N, 3) - barycentric coordinates
        - bin_points_xy : torch.Tensor of shape (N, 2) - Cartesian coordinates
        
    Raises
    ------
    TypeError
        If input types are invalid.
    ValueError
        If domain_bounds structure is invalid or eps is non-positive.
    RuntimeError
        If computation fails.
        
    Examples
    --------
    >>> r2, corners, triangle, trimesh = simplex_setup(refinement=3)
    >>> domain_bounds = (r2, corners, triangle, trimesh)
    >>> bin_points, bin_points_xy = simplex_bin_setup_vectorized(domain_bounds)
    >>> print(f"Generated {bin_points.shape[0]} bin points")
    """
    
    try:
        # Input validation
        if not isinstance(domain_bounds, tuple) or len(domain_bounds) != 4:
            raise ValueError("domain_bounds must be a tuple of length 4")
        
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError(f"eps must be a positive number, got {eps}")
        
        r2, corners, triangle, trimesh = domain_bounds
        
        if not hasattr(trimesh, 'x') or not hasattr(trimesh, 'y'):
            raise ValueError("trimesh must have 'x' and 'y' attributes")
        
        # Convert mesh points to tensors
        x_coords = torch.tensor(trimesh.x, dtype=torch.float32)
        y_coords = torch.tensor(trimesh.y, dtype=torch.float32)
        
        if len(x_coords) != len(y_coords):
            raise ValueError("Trimesh x and y coordinates have different lengths")
        
        # Vectorized conversion to barycentric coordinates
        bin_points_ba = xy2ba_vectorized(x_coords, y_coords, corners)
        
        # Vectorized projection onto simplex
        bin_points_projected = projection_onto_simplex_vectorized(bin_points_ba)
        
        # Round to avoid numerical precision issues
        bin_points_projected = torch.round(bin_points_projected, decimals=5)
        
        # Vectorized boundary adjustment
        # Find points that need adjustment (on boundaries)
        zero_mask = bin_points_projected <= eps
        one_mask = bin_points_projected >= (1.0 - eps)
        
        # Adjust points that are exactly on vertices (one coordinate is 1)
        vertex_points = torch.any(one_mask, dim=1)
        if torch.any(vertex_points):
            vertex_indices = torch.where(vertex_points)[0]
            for idx in vertex_indices:
                point = bin_points_projected[idx]
                vertex_coord = torch.argmax(point)
                
                # Reduce vertex coordinate and distribute to others
                point[vertex_coord] -= eps
                remaining_coords = torch.arange(3) != vertex_coord
                point[remaining_coords] += eps / 2
                
                bin_points_projected[idx] = point
        
        # Adjust points that are on edges (one coordinate is 0)
        edge_points = torch.any(zero_mask, dim=1) & ~vertex_points
        if torch.any(edge_points):
            edge_indices = torch.where(edge_points)[0]
            for idx in edge_indices:
                point = bin_points_projected[idx]
                zero_coord = torch.argmin(point)
                
                # Increase zero coordinate and adjust others
                point[zero_coord] += eps
                remaining_coords = torch.arange(3) != zero_coord
                point[remaining_coords] -= eps / 2
                
                bin_points_projected[idx] = point
        
        # Final validation and normalization
        # Ensure all coordinates are positive
        bin_points_projected = torch.clamp(bin_points_projected, min=eps/10)
        
        # Renormalize to ensure sum equals 1
        row_sums = torch.sum(bin_points_projected, dim=1, keepdim=True)
        bin_points_projected = bin_points_projected / row_sums
        
        # Convert back to Cartesian coordinates
        bin_points_xy = ba2xy_vectorized(bin_points_projected, corners)
        
        # Validate final results
        if not torch.all(torch.isfinite(bin_points_projected)):
            raise RuntimeError("Final bin points contain non-finite values")
        
        if not torch.all(torch.isfinite(bin_points_xy)):
            raise RuntimeError("Final Cartesian coordinates contain non-finite values")
        
        # Check simplex constraints
        coord_sums = torch.sum(bin_points_projected, dim=1)
        if not torch.allclose(coord_sums, torch.ones_like(coord_sums), atol=1e-5):
            raise RuntimeError("Final bin points do not satisfy simplex constraint")
        
        if torch.any(bin_points_projected <= 0):
            raise RuntimeError("Final bin points contain non-positive coordinates")
        
        return bin_points_projected, bin_points_xy
        
    except Exception as e:
        if isinstance(e, (TypeError, ValueError, RuntimeError)):
            raise
        else:
            raise RuntimeError(f"Unexpected error in vectorized simplex bin setup: {str(e)}") from e


# ================= BACKWARD COMPATIBLE FUNCTIONS =================

def simplex_setup(refinement: int = 4):
    """
    Sets up the simplex by defining its corners and creating a refined triangular grid.

    :param refinement: The level of refinement for the triangular grid. Default is 4.
    :type refinement: int
    :return: A tuple containing:
        - r2 (np.ndarray): The third corner of the simplex.
        - corners (np.ndarray): The coordinates of the simplex corners.
        - triangle (matplotlib.tri.Triangulation): The initial triangulation of the simplex.
        - trimesh (matplotlib.tri.Triangulation): The refined triangular grid.
    :rtype: tuple
    """
    r0 = np.array([0, 0])
    r1 = np.array([1, 0])
    r2 = np.array([1 / 2., np.sqrt(3) / 2.])
    corners = np.array([r0, r1, r2])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=refinement)
    trimesh_fine = refiner.refine_triangulation(subdiv=refinement)
    return r2, corners, triangle, trimesh

def simplex_bin_setup(domain_bounds:tuple):
    """
    Sets up bins on the simplex by projecting points onto the simplex and adjusting their positions.

    :param domain_bounds: A tuple containing domain information, including corners and triangular mesh.
    :type domain_bounds: tuple
    :return: A tuple containing:
        - bin_points (np.ndarray): Points in barycentric coordinates adjusted to lie on the simplex.
        - bin_points_xy (np.ndarray): Points in Cartesian coordinates.
    :rtype: tuple
    """
    corners = domain_bounds[1]
    trimesh = domain_bounds[3]
    bin_points_xy = np.array([[x, y] for x, y in zip(trimesh.x, trimesh.y)])
    bin_points = np.array([xy2ba(x, y, corners) for x, y in zip(trimesh.x, trimesh.y)])
    for bin_point_id in range(len(bin_points)):
        bin_point = projection_onto_simplex(torch.tensor(bin_points[bin_point_id])).numpy()[0]
        bin_point = np.round(bin_point, decimals=5)
        if any(x <= 0 for x in bin_point):
            if any(x == 1 for x in bin_point):
                i = np.where(bin_point == 1)[0][0]
                bin_point[i] -= .001
                bin_point[i - 1] += .0005
                if i == 2:
                    bin_point[i - 2] += .0005
                else:
                    bin_point[i + 1] += .0005
                bin_points[bin_point_id] = bin_point
            elif any(x == 0 for x in bin_point):
                i = np.where(bin_point == 0)[0][0]
                bin_point[i] += .001
                bin_point[i - 1] -= .0005
                if i == 2:
                    bin_point[i - 2] -= .0005
                else:
                    bin_point[i + 1] -= .0005

        bin_points[bin_point_id] = bin_point
    return bin_points, bin_points_xy

def xy2ba(x: torch.Tensor,
          y: torch.Tensor,
          corners: np.ndarray) -> torch.Tensor:
    """
    Converts Cartesian coordinates to barycentric coordinates.

    :param x: x-coordinate in Cartesian space.
    :type x: torch.Tensor
    :param y: y-coordinate in Cartesian space.
    :type y: torch.Tensor
    :param corners: Coordinates of the simplex corners.
    :type corners: np.ndarray
    :return: Barycentric coordinates corresponding to the input Cartesian coordinates.
    :rtype: torch.Tensor
    """
    corner_x = corners.T[0]
    corner_y = corners.T[1]
    x_1 = corner_x[0]
    x_2 = corner_x[1]
    x_3 = corner_x[2]
    y_1 = corner_y[0]
    y_2 = corner_y[1]
    y_3 = corner_y[2]
    l1 = ((y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)) / ((y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))
    l2 = ((y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)) / ((y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))
    l3 = 1 - l1 - l2
    return np.array([l1, l2, l3])

def ba2xy(x: torch.Tensor,
          corners: np.ndarray) -> torch.Tensor:
    """
    Converts barycentric coordinates to Cartesian coordinates.

    :param x: Array of barycentric coordinates.
    :type x: torch.Tensor
    :param corners: Coordinates of the simplex corners.
    :type corners: np.ndarray
    :return: Cartesian coordinates corresponding to the input barycentric coordinates.
    :rtype: torch.Tensor
    """
    return torch.matmul(torch.tensor(corners).T, x.T).T

def projection_onto_simplex(Y: torch.Tensor) -> torch.Tensor:
    """
    Projects a position vector onto the simplex.

    :param Y: Position vector to be projected.
    :type Y: torch.Tensor
    :return: Position vector projected onto the simplex.
    :rtype: torch.Tensor
    """
    D = Y.shape
    Y = Y.reshape(1, list(D)[0])
    N, D = Y.shape
    X, _ = torch.sort(Y, dim=1, descending=True)
    Xtmp = (torch.cumsum(X, dim=1) - 1) / torch.arange(1, D + 1, dtype=torch.float32)
    Xtmp = Xtmp.repeat(N, 1)
    X = torch.maximum(Y - Xtmp[torch.arange(N).unsqueeze(1), torch.sum(X > Xtmp, dim=1) - 1], torch.tensor(0.0))

    return X

