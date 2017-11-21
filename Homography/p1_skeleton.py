"""
CS 583 - Computer Vision 
Project #1: Homography
By Gregory Matthews
4/11/17
"""

import numpy as np

def build_A(pts1, pts2):
    """Constructs the intermediate matrix A used in the computation of an
    homography mapping pts1 to pts2."""
    # Build and return A

    # Base Case: Initialize Matrix A with first 2 rows
    v1 = np.array([pts1[0][0], pts1[0][1], 1, 0, 0, 0, -pts2[0][0] * pts1[0][0], -pts2[0][0] * pts1[0][1], -pts2[0][0]])
    v2 = np.array([0, 0, 0, pts1[0][0], pts1[0][1], 1, -pts2[0][1] * pts1[0][0], -pts2[0][1] * pts1[0][1], -pts2[0][1]])
    A = np.append([v1], [v2], axis=0)

    # Loop over the rest of pts1 and pts2, creating 2 9-element vectors for
    # each loop, and append them to A: A = 2xn*9
    for pt1, pt2 in zip(pts1[1:], pts2[1:]):
        v1 = np.array([pt1[0], pt1[1], 1, 0, 0, 0, -pt2[0]*pt1[0], -pt2[0]*pt1[1], -pt2[0]])
        v2 = np.array([0, 0, 0, pt1[0], pt1[1], 1, -pt2[1]*pt1[0], -pt2[1]*pt1[1], -pt2[1]])
        A = np.append(A, [v1], axis=0)
        A = np.append(A, [v2], axis=0)

    return A


def compute_H(pts1, pts2):
    """Computes a homography mapping one set of co-planar points (pts1) to another (pts2)."""
    
    # Construct the intermediate A matrix.
    A = build_A(pts1, pts2)

    # Compute the symmetric matrix AtA.
    A_t = A.transpose()
    H = np.dot(A_t, A)

    # Compute the eigenvalues and eigenvectors of AtA.
    w, v = np.linalg.eigh(H)
    
    # Return the eigenvector corresponding to the smallest eigenvalue, reshaped
    # as a 3x3 matrix.

    # find smallest eigenvalue
    min_value = np.amin(w)

    # find index of corresponding eigenvector
    #vector_index = np.where(w == min_value)[0]
    vector_index = 0

    # retrieve eigenvector and transform into 3x3 matrix
    v = np.transpose(v)
    min_v = v[:][vector_index]
    eig_matrix = np.reshape(min_v, (3,3))
  
    return eig_matrix


def bilinear_interp(image, point):
    """Looks up the pixel values in an image at a given point using bilinear
    interpolation."""
    
    # Compute the four integer corner coordinates (top-left/right,
    # bottom-left/right) for interpolation.
    top_l = image[int(np.ceil(point[1]))][int(np.floor(point[0]))]
    top_r = image[int(np.ceil(point[1]))][int(np.ceil(point[0]))]
    bottom_l = image[int(np.floor(point[1]))][int(np.floor(point[0]))]
    bottom_r = image[int(np.floor(point[1]))][int(np.ceil(point[0]))]

    # Compute the fractional part of the point coordinates.
    a = np.modf(point[0])[0]
    b = np.modf(point[1])[0]

    # Interpolate between the top two pixels.
    R1 = np.add([a * x for x in top_l], [(1 - a) * x for x in top_r])

    # Interpolate between the bottom two pixels.
    R2 = np.add([a * x for x in bottom_l], [(1 - a) * x for x in bottom_r])

    # Return the result of the final interpolation between top and bottom.
    P = np.add([b * x for x in R1], [(1 - b) * x for x in R2])

    return P

def warp_homography(source, target_shape, Hinv):
    """Warp the source image into the target coordinate frame using a provided
    inverse homography transformation."""

    #   Steps for Warping:
    #   1.) Transforming points in output image to source image space  
    #   2.) Look up pixel value in source image at transformed points
    
    # Create coordinate matrix of output plane using mgrid
    col, row = np.mgrid[0:target_shape[1], 0:target_shape[0]]
    output_image = np.empty([int(target_shape[1]), int(target_shape[0]), 3]) 
   
    # Iterate over coordinate plane of target image
    for x1, y1 in zip(row, col):
        for x2, y2 in zip(x1, y1):
    
            # Calculate source points using homography
            new_pt = np.dot(Hinv, [x2, y2, 1])
            new_pt = np.divide(new_pt, new_pt[2])
            
            # Check if index is negative after offsetting; fell out of bounds
            if new_pt[0] < 0 or new_pt[1] < 0:
                pixel = [0,0,0]
            
            else:
                # Attempt indexing of source image for pixel value
                try: 
                    pixel = bilinear_interp(source, [new_pt[0], new_pt[1]])
            
                # If index out of bounds, generate a black pixel
                except IndexError:
                    pixel = [0, 0, 0]
            
                # Update pixel value in output image array
                output_image[int(y2)][int(x2)] = pixel 
    
    return output_image



def rectify(image, planar_points, target_points):
    # Compute the rectifying homography that warps the planar points to the 
    # target points
    H = compute_H(planar_points, target_points)
   
    # Apply the homography to the bounding box of the planar image to find the 
    # warped bounding box in the rectified space
    size = image.shape
    planar_bounds = np.matrix([[0, 0, 1], [size[1], 0, 1], [size[1], size[0], 1], [0, size[0], 1]])     
    warped_bounds = np.dot(H, np.transpose(planar_bounds))

    # divide [x,y,z] homogonous coordinates by z to get (x,y)
    z =  warped_bounds[2]
    warped_bounds = np.divide(warped_bounds, z)

    # Offset the warped bounding box so that none of its points contain 
    # negative X/Y coordinates
    x = -np.amin(warped_bounds[0])
    y = -np.amin(warped_bounds[1])
  
    if x < 0: x = 0
    if y < 0: y = 0
 
    # Compute new output coordinate bounds from offset
    x_cords = np.add(x, warped_bounds[0])
    y_cords = np.add(y, warped_bounds[1])
 
    height = np.amax(y_cords)
    width = np.amax(x_cords)

    # Create target shape
    target_shape = np.array([np.ceil(width),np.ceil(height)])

    # Compute the inverse homography to warp between the offset, warped
    # bounding box and the bounding box of the input image.
    warped_bounds = np.concatenate([x_cords, y_cords], axis=0)
    warped_bounds = np.transpose(warped_bounds)
    planar_bounds = np.delete(planar_bounds, 2, 1)

    H_inv = compute_H(np.asarray(warped_bounds), np.asarray(planar_bounds))   

    # Perform inverse warping and return the result.
    output_image =  warp_homography(image, target_shape, H_inv)
   
    return output_image

def blend_with_mask(source, target, mask):
    """Blends the source image with the target image according to the mask.
    Pixels with value "1" are source pixels, "0" are target pixels, and
    intermediate values are interpolated linearly between the two.""" 
    size = mask.shape
    target_shape = [size[1], size[0]]

    # Create coordinate matrix of output plane using mgrid
    col, row = np.mgrid[0:target_shape[1], 0:target_shape[0]]
    output_image = np.empty([int(target_shape[1]), int(target_shape[0]), 3]) 
   
    # Iterate over coordinate plane of target image
    for x1, y1 in zip(row, col):
        for x2, y2 in zip(x1, y1):

            pixel = np.int8(mask[y2][x2])
            flag = int(pixel[0])
            flag = int(mask[y2][x2][0])
            
            if flag == 1: #retrieve source pixels
                pixel = source[y2][x2]
            
            elif flag == 0: #retrieve target pixels
                pixel = target[y2][x2]
            
            else: #Linear Interpolation

                # Compute the fractional part of the point coordinates.
                a = np.modf(y2)

                # Interpolate between the top two pixels.
                R1 = np.add([a * x for x in source[y2][x2]], [(1 - a) * x for x in target[y2][x2]])

            # Update pixel value in output image array
            output_image[int(y2)][int(x2)] = pixel 
    
    return output_image



    return 1

def composite(source, target, source_pts, target_pts, mask):
    """Composites a masked planar region of the source image onto a
    corresponding planar region of the target image via homography warping."""
    # Compute the homography to warp points from the target to the source coordinate frame.
    H_inv = compute_H(source_pts, target_pts)
   
    # Warp the correct images using the homography.
    size = source.shape 
    warped_image = warp_homography(target, [size[1],size[0]], H_inv)
    warped_mask = warp_homography(mask, [size[1], size[0]], H_inv)

    # Blend the warped images and return them.
    output_image = blend_with_mask(warped_image, source, warped_mask)

    return output_image


def mosaicing(img1, img2, img3, pts1_3, pts3_1, pts2_3, pts3_2):
    
    H = compute_H(pts1_3, pts3_1)
    H_inv = np.linalg.inv(H) 

    warped_img = img3
    #warped_img = warp_homography(img1, [size[1],size[0]], H_inv)
    #warped_img = rectify(img1, pts1_3, pts3_1)
   
    # Apply the homography to the bounding box of the planar image to find the 
    # warped bounding box in the rectified space
    size = img1.shape
    planar_bounds = np.matrix([[0, 0, 1], [size[1], 0, 1], [size[1], size[0], 1], [0, size[0], 1]]) 
    warped_bounds = np.dot(H, np.transpose(planar_bounds))
    warped_bounds = np.transpose(warped_bounds)

    # divide [x,y,z] homogonous coordinates by z to get (x,y)
    z =  np.transpose(warped_bounds)[2]
    z = np.transpose(z)
    warped_bounds = np.divide(warped_bounds, z)

    # Offset the warped bounding box so that none of its points contain 
    # negative X/Y coordinates
    warped_bounds = np.transpose(warped_bounds)
    x = -np.amin(warped_bounds[0])
    y = -np.amin(warped_bounds[1])
    
    # Check if no offset is needed
    """
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    """

    # Warp points mapping img 1 to 3
    planar_pts = np.transpose(pts1_3)
    planar_pts = np.append(planar_pts, [1,1,1,1])
    planar_pts = planar_pts.reshape((3,4)) 
    warped_pts = np.dot(H, planar_pts)

    print warped_pts

    # divide [x,y,z] homogonous coordinates by z to get (x,y)
    z =  warped_pts[2]
    warped_pts = np.divide(warped_pts, z)
    warped_pts = np.transpose(warped_pts)

    print warped_pts

    #composite(warped_image, img3, warped_pts, pts3_1, 


    warped = np.transpose(pts3_1)
    
    size1 = img1.shape

    new_y = np.add(warped[1], size[1]-166)
    new_x = warped[0]
    warped = np.append([new_x], [new_y], axis=0)
    warped = np.transpose(warped)
   
    print pts1_3
    print warped

    H2 = compute_H(warped, pts3_1) 
    warped_img = warp_homography(img3, [573, 476], H2)
  



    return warped_img



