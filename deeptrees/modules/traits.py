import pandas as pd
import shapely.wkb as wkb
from shapely.geometry import Polygon
from shapely import LineString, Point
import math
import numpy as np


# Calculate the crown spread for each polygon
# Function to calculate the longest spread
def longest_spread(polygon):
    """
    Calculate the longest distance between any two points on the exterior of a polygon.
    Parameters:
    polygon (shapely.geometry.Polygon): A shapely Polygon object. If the polygon is None or not a valid polygon, the function returns (None, None).
    Returns:
    tuple: A tuple containing the maximum distance (float) and the pair of points (tuple of shapely.geometry.Point) that are farthest apart. 
           If the polygon is invalid, returns (None, None).
    """
    
    # Check if the polygon is None or not a valid polygon
    if polygon is None or not hasattr(polygon, 'exterior'):
        return None, None 
    
    # Extract the exterior points of the polygon
    points = list(polygon.exterior.coords)
    max_distance = 0
    point_pair = None

    # Calculate the maximum distance between any two points in the exterior
    for i, point1 in enumerate(points):
        for point2 in points[i+1:]:
            dist = Point(point1).distance(Point(point2))
            if dist > max_distance:
                max_distance = dist
                point_pair = (Point(point1), Point(point2))
    
    return max_distance, point_pair

# Function to calculate the longest cross-spread
def longest_cross_spread(polygon):
    """
    Calculate the longest cross spread of a given polygon.
    The longest cross spread is defined as the maximum distance between two points
    on the polygon's boundary that are perpendicular to the longest spread line of the polygon.
    Parameters:
    polygon (shapely.geometry.Polygon): The input polygon for which the longest cross spread is calculated.
    Returns:
    tuple: A tuple containing:
        - max_cross_distance (float): The maximum cross distance found.
        - cross_point_pair (tuple): A tuple of two shapely.geometry.Point objects representing the endpoints of the longest cross spread.
          If the polygon is None or invalid, or if no valid cross spread is found, returns (None, None).
    """
    
    # Check if the polygon is None or invalid
    if polygon is None or not hasattr(polygon, 'exterior'):
        return None, None  

    # Get the longest spread and endpoints
    longest_spread_result = longest_spread(polygon)
    if longest_spread_result[0] is None or longest_spread_result[1] is None:
        return None, None  

    max_distance, (point1, point2) = longest_spread_result
    
    # Define the longest spread line and calculate the perpendicular direction
    spread_line = LineString([point1, point2])
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    perpendicular_angle = math.atan2(dy, dx) + math.pi / 2  # 90-degree rotation
    
    # Initialize maximum cross distance variables
    max_cross_distance = 0
    cross_point_pair = None
    
    # Iterate over all boundary points and measure perpendicular distances
    points = list(polygon.exterior.coords)
    for i, point in enumerate(points):
        candidate_point = Point(point)
        
        # Find the direction vector perpendicular to the spread line
        perp_dx = math.cos(perpendicular_angle)
        perp_dy = math.sin(perpendicular_angle)
        
        # Extend the candidate point in both positive and negative perpendicular directions
        max_proj_point = None
        max_proj_distance = 0
        
        for sign in [-1, 1]:  # Check both directions
            extended_point = Point(
                candidate_point.x + sign * perp_dx * 1000,  # Extend sufficiently far in each direction
                candidate_point.y + sign * perp_dy * 1000
            )
            
            # Find the intersection of the extended line with the polygon boundary
            perp_line = LineString([candidate_point, extended_point])
            intersection = perp_line.intersection(polygon.exterior)
            
            # Calculate distance if there is an intersection point
            if not intersection.is_empty:
                if intersection.geom_type == "MultiPoint":
                    # If multiple points intersect, find the farthest point
                    for pt in intersection.geoms:
                        dist = candidate_point.distance(pt)
                        if dist > max_proj_distance:
                            max_proj_distance = dist
                            max_proj_point = pt
                elif intersection.geom_type == "Point":
                    dist = candidate_point.distance(intersection)
                    if dist > max_proj_distance:
                        max_proj_distance = dist
                        max_proj_point = intersection
        
        # Update the maximum cross distance if found
        if max_proj_distance > max_cross_distance:
            max_cross_distance = max_proj_distance
            cross_point_pair = (candidate_point, max_proj_point)
    
    return max_cross_distance, cross_point_pair

def chlorophyll_index(red, green, nir, save_to=None):
    """
    Calculate the Chlorophyll Index-Green (CIG) from the given bands.
    The Chlorophyll Index Green (CIG) is defined as (NIR/Green)-1.
    Parameters:
    red (numpy.ndarray): The red band values.
    green (numpy.ndarray): The green band values.
    nir (numpy.ndarray): The near-infrared (NIR) band values.
    save_to (str): The file path to save the calculated GCI values as a CSV file. Default is None.
    Returns:
    numpy.ndarray: The calculated  Chlorophyll Index (CIG) values.
    """
    cig = (nir.astype(float) / green.astype(float) ) - 1
    if save_to:
        df = pd.DataFrame(cig)
        df.to_csv(save_to, index=False, header=False)
    return cig

def green_chlorophyll_index(red, green, nir, save_to=None):
    """
    Calculate the Green Chlorophyll Index (GCI) from the given bands.
    The Green Chlorophyll Index (GCI) is defined as (NIR-Green)/(NIR+Green).
    Parameters:
    red (numpy.ndarray): The red band values.
    green (numpy.ndarray): The green band values.
    nir (numpy.ndarray): The near-infrared (NIR) band values.
    save_to (str): The file path to save the calculated GCI values as a CSV file. Default is None.
    Returns:
    numpy.ndarray: The calculated Green Chlorophyll Index (GCI) values.
    """
    gci = (nir.astype(float) - green.astype(float)) / (nir.astype(float) + green.astype(float))
    if save_to:
        df = pd.DataFrame(gci)
        df.to_csv(save_to, index=False, header=False)
    return gci

def hue_index(red, green, blue, save_to=None):
    """
    Calculate the Hue Index from the given bands.
    The Hue Index is defined as arctan((2*green - red - blue) / sqrt(3) * (red - blue)).
    Parameters:
    red (numpy.ndarray): The red band values.
    green (numpy.ndarray): The green band values.
    blue (numpy.ndarray): The blue band values.
    save_to (str): The file path to save the calculated Hue Index values as a CSV file. Default is None.
    Returns:
    numpy.ndarray: The calculated Hue Index values.
    """
    hue = np.arctan((2 * green - red - blue) / (np.sqrt(3) * (red - blue)))
    if save_to:
        df = pd.DataFrame(hue)
        df.to_csv(save_to, index=False, header=False)
    return hue