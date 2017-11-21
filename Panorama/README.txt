Objective:   
Step 1.) Convert the captured images of the panorama from planar coordinates to cylindrical coordinates    
Step 2.) Compute the displacement between 2 images by estimating the images optical flow using Lukas Kanade  
Step 3.) Estimate the optical flow more efficiently by scaling down the image size until it is less than a pixel,     
         using Gaussian blending each ½ size decrementation from the original size.     
Step 4.) Estimate the refined optical flow by using both the Gaussian pyramid and Lucas Kanade implementations.   
Step 5.) From running the Pyramid Lucas Kanade implementation and receiving the final displacements, build the panorama by aligning images correctly onto a blank canvas. 
