************************************************
	Created by:  Leevon Levasseur
	Last Update: April 16, 2020
	Title:       RayTraced Image

************************************************

RayTraced Image

This ray traced image was coded in C++. The multi-sampled, anti-aliased image contains multiple objects with different materials, multiple lights, shadows, and a pin-hole camera 


------------------------------------------------

		   Objects

------------------------------------------------

Objects included:

	- Spheres
	- Triangle
	- Plane


------------------------------------------------

	       Pin-hole Camera

------------------------------------------------

The pin-hole camera can be coded anywhere in 3D space. The "eye" is where the camera is looking from, the "look at" is where the camera is looking, and the "up" is the directional vector
above the camera.


------------------------------------------------

		   Lights

------------------------------------------------

Lights included:
	
	- Ambient Occlusion
	- Directional
	- Point


------------------------------------------------

	   Multi-sampling Technique

------------------------------------------------

I chose to use multi-sampling to render my image. This allows edges of objects seem sharp by using a blend of colours for pixels on the edges. This process is as follows: for each pixel,
shoot multiple rays at varying locations of the pixel, then take the colours of all objects hit and find the average colour. This is the resulting colour. If a ray does not hit an object
the resulting colour is that of the background.


------------------------------------------------

		  Materials

------------------------------------------------

Using complex calculus, Bidirectional Reflectance Distribution Functions (BRDF's) and Bidirectional Transmittance Distribution Functions (BTDF's) were used to make objects seem like
they are made of different materials.

Included materials:
	
	- Matte
	- Glossy (Phong)
	- Reflective
	- Simple Transparency


************************************************
	
	 Final Render is found in this
	 repository with the name
	 raytrace.bmp at 1000x1000
	 resolution.

************************************************








