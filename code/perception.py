import numpy as np
import cv2
from supporting_functions import MASKDRAWING , SOURCE , DEST , SCALE , MASKDRAWING_3D ,MASKMOVING


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(155 , 140 , 127)): # (155 , 140 , 127) (160 , 160 , 160)
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def color_rocks(img):
    # Create an array of zeros same xy size as img, but single channel
    lower_yellow = np.array([24-10,100,100])
    upper_yellow = np.array([24+10,255,255])

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Threshold the HSV image to get only upper_yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Return the binary image
    return mask

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def getobstacles(threshed,mask):
    obstacles = np.absolute((np.float32(threshed)-1) * mask)
    return obstacles

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    wraped = perspect_transform(Rover.img,SOURCE,DEST)
    # threshed = color_thresh(wraped)
    drawing_threshed = color_thresh(wraped)*MASKDRAWING 
    moving_threshed = color_thresh(wraped,(160,160,160))*MASKMOVING
    obstacles = getobstacles(drawing_threshed,MASKDRAWING)
    rock_samples = color_rocks(wraped)*MASKDRAWING
    
    vision_image = (wraped*MASKDRAWING_3D).astype(np.uint8)
    # (160, 320, 3)
    Rover.vision_image = vision_image #wraped

    xpix_mov , y_pix_mov = rover_coords(moving_threshed)

    xpix_navigable, ypix_navigable = rover_coords(drawing_threshed)
    xpix_obstacles, ypix_obstacles = rover_coords(obstacles)
    xpix_rocks, ypix_rocks = rover_coords(rock_samples)

    navigable_x_world, navigable_y_world = pix_to_world(xpix_navigable, ypix_navigable, 
                                                        Rover.pos[0], Rover.pos[1], 
                                                        Rover.yaw, Rover.worldmap.shape[0], SCALE)
    obstacle_x_world, obstacle_y_world = pix_to_world(xpix_obstacles, ypix_obstacles, 
                                                        Rover.pos[0], Rover.pos[1], 
                                                        Rover.yaw, Rover.worldmap.shape[0], SCALE)
    rock_x_world, rock_y_world = pix_to_world(xpix_rocks, ypix_rocks, 
                                                Rover.pos[0], Rover.pos[1], 
                                                Rover.yaw, Rover.worldmap.shape[0], SCALE)

    if (Rover.pitch < 1 or Rover.pitch > 359) and (Rover.roll < 1 or Rover.roll > 359) and (Rover.brake <= 0):
        distance_obstacles = np.sqrt(np.power(obstacle_x_world-Rover.pos[0] ,2)+np.power(obstacle_y_world-Rover.pos[1] ,2)).astype(np.uint8)
        distance_navigable = np.sqrt(np.power(navigable_x_world-Rover.pos[0] ,2)+np.power(navigable_y_world-Rover.pos[1] ,2)).astype(np.uint8)
        
        if distance_obstacles.size > 0 and  distance_navigable.size > 0:
            val = max(distance_obstacles.max(),distance_navigable.max())
            Rover.vote[obstacle_y_world, obstacle_x_world, 0] += (val-distance_obstacles)
            Rover.vote[navigable_y_world, navigable_x_world, 2] += (val+10-distance_navigable) 
            nav_pix = (Rover.vote[:,:,2] > Rover.vote[:,:,0])
            Rover.worldmap[nav_pix,2] = 255
            Rover.worldmap[nav_pix,0] = 0
            Rover.worldmap[~nav_pix,0] = 255
            Rover.worldmap[~nav_pix,2] = 0
            Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
        
    Rover.nav_dists , Rover.nav_angles = to_polar_coords(xpix_mov, y_pix_mov)
    if xpix_rocks.size > 0 and ypix_rocks.size > 0:    
        dist, angles = to_polar_coords(xpix_rocks, ypix_rocks)
        Rover.samples_dists = dist
        Rover.samples_angles = angles
        Rover.wrong_rock = 0
    elif Rover.wrong_rock > 5:
        Rover.samples_dists = None
        Rover.samples_angles = None
    elif Rover.samples_dists is not None and Rover.samples_angles is not None: 
        Rover.wrong_rock += 1

    # clip to avoid overflow
    Rover.worldmap = np.clip(Rover.worldmap, 0, 255)
    
    return Rover