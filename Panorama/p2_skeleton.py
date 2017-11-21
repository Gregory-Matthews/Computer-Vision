"""
CS 583: Project 2 - Panorama
By Greg Matthews
5/14/17
"""

import pickle
import numpy as np
from scipy.ndimage.filters import convolve


def bilinear_interp(image, points):
    """Given an image and an array of row/col (Y/X) points, perform bilinear
    interpolation and return the pixel values in the image at those points."""
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis]

    valid = np.all(points < [image.shape[0]-1, image.shape[1]-1], axis=-1)
    valid *= np.all(points >= 0, axis=-1)
    valid = valid.astype(np.float32)
    points = np.minimum(points, [image.shape[0]-2, image.shape[1]-2])
    points = np.maximum(points, 0)

    fpart, ipart = np.modf(points)
    tl = ipart.astype(np.int32)
    br = tl+1
    tr = np.concatenate([tl[..., 0:1], br[..., 1:2]], axis=-1)
    bl = np.concatenate([br[..., 0:1], tl[..., 1:2]], axis=-1)

    b = fpart[..., 0:1]
    a = fpart[..., 1:2]

    top = (1-a) * image[tl[..., 0], tl[..., 1]] + a * image[tr[..., 0], tr[..., 1]]
    bot = (1-a) * image[bl[..., 0], bl[..., 1]] + a * image[br[..., 0], br[..., 1]]
    return ((1-b) * top + b * bot) * valid[..., np.newaxis]

def translate(image, displacement):
    """Takes an image and a displacement of the form X,Y and translates the
    image by the displacement. The shape of the output is the same as the
    input, with missing pixels filled in with zeros."""
    pts = np.mgrid[:image.shape[0], :image.shape[1]].transpose(1,2,0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)

def to_cylindrical(image, camera_params):
    F, k1, k2 = camera_params
    cyl_img_pts = np.mgrid[:image.shape[0], :image.shape[1]].transpose(1,2,0).astype(np.float32)
    x_cyl = cyl_img_pts[..., 1]
    y_cyl = cyl_img_pts[..., 0]

    # Convert from cylindrical image points x_cyl/y_cyl to cylindrical
    # coordinates theta/h.
    xc = image.shape[1]/2.
    yc = image.shape[0]/2.
    theta = (x_cyl - xc)/F
    h = (y_cyl - yc)/F

    # Convert from cylindrical coordinates theta/h to x/y/z coordinates on the
    # 3D cylinder.
    x_hat = np.sin(theta)
    y_hat = h
    z_hat = np.cos(theta)

    # Convert from cylinder x/y/z to normalized input image x/y coordinates.
    x_norm = x_hat/z_hat
    y_norm = y_hat/z_hat

    # Apply radial distortion correction to the normalized image x/y
    # coordinates.
    r2 = np.power(x_norm,2) + np.power(y_norm,2)
    x_dist = x_norm*(1 + k1*r2 + k2*np.power(r2, 2))
    y_dist = y_norm*(1 + k1*r2 + k2*np.power(r2,2))

    # Convert from normalized image x/y coordinates to actual x/y coordinates.
    x_prime = F*x_dist + xc
    y_prime = F*y_dist + yc

    # Look up pixels in the input i at the final coordinates (using bilinear
    # interpolation) to form the cylindrical image and return it.
    points = np.column_stack((y_prime.flatten(), x_prime.flatten()))
    points = points.reshape((image.shape[0], image.shape[1], 2))
    image_cyl = bilinear_interp(image, points).astype(np.float32)

    return image_cyl

def lucas_kanade(H, I):
    """Given images H and I, compute the displacement that should be applied to
    H so that it aligns with I."""
    # Generate a binary mask indicating pixels that are valid (non-black) in
    # both H and I.
    bin_mask = np.all((I > 0), axis=-1)*np.all((H > 0), axis=-1)

    # AND the mask with another mask that selects pixels that aren't dark
    # (intensity > 0.25).
    tol_mask = (np.average(I,2) > 0.25) * (np.average(H,2) > 0.25)
    and_mask = bin_mask * tol_mask
   
    # Compute the partial image derivatives w.r.t. X, Y, and Time (t).
    It = (I - H)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)/8.
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)/8.

    Ix1 = convolve(I[:,:,0], sobel_x) 
    Iy1 = convolve(I[:,:,0], sobel_y)  

    Ix2 = convolve(I[:,:,1], sobel_x) 
    Iy2 = convolve(I[:,:,1], sobel_y) 
 
    Ix3 = convolve(I[:,:,2], sobel_x) 
    Iy3 = convolve(I[:,:,2], sobel_y) 

    Ix = np.dstack([Ix1, Ix2, Ix3])
    Iy = np.dstack([Iy1, Iy2, Iy3])

    # Compute the various products (Ixx, Ixy, Iyy, Ixt, Iyt) necessary to form
    # AtA. Apply the mask to each product to select only valid values.
    Ixx = np.sum((Ix * Ix) * and_mask[..., np.newaxis])
    Ixy = np.sum((Ix * Iy) * and_mask[..., np.newaxis])
    Iyy = np.sum((Iy * Iy) * and_mask[..., np.newaxis])
    Ixt = np.sum((Ix * It) * and_mask[..., np.newaxis])
    Iyt = np.sum((Iy * It) * and_mask[..., np.newaxis])

    # Build the AtA matrix and Atb vector
    AtA = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    Atb = np.array([[Ixt],[Iyt]])

    # Solve the system and return the computed displacement.
    displacement = np.linalg.solve(AtA, -Atb)
    #displacement = np.linalg.lstsq(AtA,-Atb)[0]
    displacement = displacement.reshape((2,))

    return displacement

def iterative_lucas_kanade(H, I, steps):
    # Run the basic Lucas Kanade algorithm in a loop `steps` times.
    # Start with an initial displacement of 0 and accumulate displacements.
    displacement = np.zeros((2,))
    H_img, I_img  = np.copy(H), np.copy(I)
    
    for i in range(steps):
        # Translate the input image by the current displacement, then run Lucas
        # Kanade and update the displacement. 
        I_img = translate(I, -displacement)
        displacement += lucas_kanade(H_img, I_img)

    # Return the final displacement
    return displacement

def gaussian_pyramid(image, levels):
    # Build a Gaussian pyramid for an image with the given number of levels,
    # then return it. 
    pyramid = []
    
    # Creating Gaussian Kernel
    sigma, size_x, size_y = 1., 5., 5.
    x, y = np.mgrid[-size_y:size_y, -size_x:size_x].astype(np.float32)
    gauss_kernel = (1/(2*np.pi*sigma**2.))*np.exp(-((x**2+y**2)/(2.*sigma**2)))
 
    for level in range(levels):
        # Convolve image with gauss kernel, do for each RGB value
        c1 = convolve(image[:,:,0], gauss_kernel) 
        c2 = convolve(image[:,:,1], gauss_kernel)
        c3 = convolve(image[:,:,2], gauss_kernel)
        image = np.dstack((c1, c2, c3))
        pyramid.insert(0, image)     
       
        # Subsample
        image = image[::2, ::2]

    return pyramid

def pyramid_lucas_kanade(H, I, initial_d, levels, steps):
    """Given images H and I, and an initial displacement that roughly aligns H
    to I when applied to H, run Iterative Lucas Kanade on a pyramid of the
    images with the given number of levels to compute the refined
    displacement."""

    # Build Gaussian pyramids for the two images.
    H_pyramid = gaussian_pyramid(H, levels)
    I_pyramid = gaussian_pyramid(I, levels) 

    # Start with the initial displacement, scaled to the coarsest level of the
    # pyramid, and compute the updated displacement at each level using
    # Iterative Lucas Kanade. 
    global_displacement = initial_d / 2.**(levels)
    
    for level in range(levels):
        # Get the two images for this pyramid level.
        H_img = H_pyramid[level]
        I_img = I_pyramid[level]
       
        # Scale the previous level's global displacement properly and apply it
        # to the second image (translate the image). 
        global_displacement *= 2.
        I_img  = translate(I_img, -global_displacement)  
    
        # Use the iterative Lucas Kanade method to compute a displacement
        # between the two images at this level.
        displacement = iterative_lucas_kanade(H_img, I_img, steps)
        
        # Update the global displacement based on the one you just computed.
        global_displacement += displacement
    
    # Return the final displacement.
    return global_displacement

def build_panorama(images, shape, displacements, initial_position, blend_width=64):
    # Allocate an empty floating-point image with space to store the panorama
    # with the given shape.
    panorama = np.zeros((shape))
    pan_mask = np.zeros((shape))
    pan_target = np.zeros((shape))
    
    # Place the last image, warped to align with the first, at its proper place
    # to initialize the panorama.
    last = images[-1][0:images[-1].shape[0], blend_width:(images[-1].shape[1]-blend_width)]
    height= last.shape[0]
    width = last.shape[1]
    u = initial_position[0] - displacements[-1][0]
    v = initial_position[1] - displacements[-1][1]

    last = translate(last,  -displacements[-1])
    panorama[int(v):height+int(v), int(u):width+int(u)] = last
    
    # Creating Linear Blend Mask
    mask = np.ones((height, width))
    for j in range(height):
        weight = 0
        for i in range(width): 
            if i <= blend_width: 
                mask[j][i] = weight
                weight += 1./blend_width
 
    # Place the images at their final positions inside the panorama, blending
    # each image with the panorama in progress. Use a blending window with the
    # given width.
    N = len(images)

    for i in range(len(images)):
        # Cropping Artifacts some cylindrical warping on sides
        images[i] = images[i][0:images[i].shape[0], blend_width:(images[i].shape[1]-blend_width)]
        
        # Linear Blending
        pan_mask[int(v):height+int(v), int(u):width+int(u)] = mask[..., np.newaxis]
        pan_target[int(v):height+int(v), int(u):width+int(u)] = images[i]
        panorama = blend_with_mask(panorama, pan_target, pan_mask)

        # Incremenet displacements
        u -= displacements[i][0]
        v -= displacements[i][1]

    # Return the finished panorama.
    return panorama

# Displacements are by default saved to a file after every run. These flags
# control loading and saving of the displacements. Once you have confirmed your
# LK code is working, you can load saved displacements to save time testing the
# rest of the project.
SAVE_DISPLACEMENTS = True
LOAD_DISPLACEMENTS = True

def mosaic(images, initial_displacements):
    """Given a list of N images taken in clockwise order and corresponding
    initial X/Y displacements of shape (N,2), refine the displacements and
    build a mosaic.
    initial_displacement[i] gives the translation that should be appiled to
    images[i] to align it with images[(i+1) % N]."""
    N = len(images)
    if LOAD_DISPLACEMENTS:
        print("Loading saved displacements...")
        final_displacements = pickle.load(open("final_displacements.pkl", "rb"))
    else:
        print("Refining displacements with Pyramid Iterative Lucas Kanade...")
        final_displacements = []
        for i in range(N):
            # Use Pyramid Iterative Lucas Kanade to compute displacements from
            # the end. A suggested number of levels and steps is 4 and 5
            # respectively. Make sure to append the displacement to
            # final_displacements so it gets saved to disk if desired.
            displacement = pyramid_lucas_kanade(images[i], images[(i+1) % N], initial_displacements[i], 4, 5)
            final_displacements.append(displacement)
   
            # Some debugging output to help diagnose errors.
            print("Image %d:" % i,
                    initial_displacements[i], "->", final_displacements[i], "  ",
                    "%0.4f" % abs((images[i] - translate(images[(i+1) % N], -initial_displacements[i]))).mean(), "->",
                    "%0.4f" % abs((images[i] - translate(images[(i+1) % N], -final_displacements[i]))).mean())

        if SAVE_DISPLACEMENTS:
            pickle.dump(final_displacements, open("final_displacements.pkl", "wb"))

    # Use the final displacements and the images' shape compute the full
    # panorama shape and the starting position for the first panorama image.
    #final_displacements = np.array([[-259.3, 0.394], [-258.8, 2.109], [-274.93, 5.567], [-280.43, 3.44], [-274.90, 7.0525], [-305.57, 8.793], [-314.36, 3.933], [-341.20, 5.45], [-306.66, 1.24], [-301.6, 8.73], [-339.78, -1.007], [-319.65, 4.34], [-376.85, 8.074], [-302.002, 4.174], [-246.2, 4.98]])
    
    width = images[0].shape[1]
    height = images[0].shape[0]
    x_disp = np.transpose(final_displacements)[0].flatten()
    y_disp = np.transpose(final_displacements)[1].flatten()
   
    # Finding y bounds
    y_sum, y_max, y_min = 0, 0, 0
    for y in y_disp:
        y_sum += y
        if y_sum > y_max:
            y_max = y_sum
        if y_sum < y_min:
            y_min = y_sum

    # Create Initial Position
    y_max = int(np.ceil(y_max)) 
    init_pos = [x_disp[-1], y_disp[-1] + np.ceil(y_max)]
    
    # Create Panorama Shape
    #pano_width = width*(N) + np.sum(np.ceil(x_disp)) + x_disp[0] 
    pano_width = width + np.abs(np.sum(np.ceil(x_disp)))
    pano_height = height + np.abs(np.ceil(y_min)) + np.abs(np.ceil(y_max))
    shape = [int(pano_height), int(pano_width), 3]
    
    # Build the panorama.
    print("Building panorama...")
    panorama = build_panorama(images, shape, final_displacements, init_pos, blend_width=32)

    # Resample the panorama image using a linear warp to distribute any vertical
    # drift to all of the sampling points. The final height of the panorama should
    # be equal to the height of one of the images.
    print("Warping to correct vertical drift...")

    y,x = np.mgrid[:shape[0], :shape[1]]
    drift = y_max + y_min
    adjust = int(pano_width/drift)
    cum_adjust = 0

    # Choose drift adjustment direction
    if drift > 0: direction = 1
    else: direction = -1

    # Perform global warp on each column
    for i in range(int(pano_width)):
        if (i % adjust == 0):
            cum_adjust += direction
        y[:,i] -= cum_adjust
    
    panorama = bilinear_interp(panorama, np.array([y,x]).transpose(1,2,0)) 
    panorama = panorama[int(y_max):height+int(y_max), :] 
  
    # Crop the panorama horizontally so that the left and right edges of the
    # panorama match (making it form a loop).
    cropped_width = panorama.shape[1]-width
    panorama = panorama[0:panorama.shape[0], 0:cropped_width]

    # Return your final corrected panorama.
    return panorama

def blend_with_mask(source, target, mask):
    masked_source = source * (1-mask)
    masked_target = target * mask
    return masked_source + masked_target

