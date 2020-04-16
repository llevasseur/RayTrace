**********************************************
	Created by:  Leevon Levasseur

	Last Update: April 16, 2020

	Title:       RayTraced Image

**********************************************

RayTraced Image


This ray traced image is being coded in C++. Using cmake and a built in library called Atlas, I have created a multisampled, anti-aliased image that contains multiple objects of different materials and sizes
on a ground plane, with lights and shadows, and a pin-hole camera. 


----------------------------------------------

		Objects

----------------------------------------------

Objects include:
	
	- Spheres
	- Triangle
	- Plane


----------------------------------------------

		 Lights

----------------------------------------------

Lights include:

	- Ambient Oclusion
	- Directional Light
	- Point Lights


----------------------------------------------

	     Pin-Hole Camera

----------------------------------------------

The pin-hole camera is a camera that can be coded to exist anywhere in 3D space. The idea is that its eye is the size of a hole a pin would make in space, and looks directly at a 
specific point. Using these two locations and the "up" direction, a coordinate system can be created for the camera, and the objects that exist in the "world coordinates" can be
represented using the camera's coordinates.


----------------------------------------------

	    Sampling Technique

----------------------------------------------


I chose to use a multisampling technique to render my image. This means for each pixel, sum all the colours of objects hit by the ray tracer. Then take the average and use that colour
as the final colour of the pixel. This gives the image sharper edges by making the pixelated edges blend with the background or other objects. I chose to use the multi-jittered technique
which uses both regular jittered and n-rook multisampling techniques. This is also references as an anti-aliasing effect.


----------------------------------------------

		Materials

----------------------------------------------


Using complex mathematics, I have created a verity of Bidirectional Reflectance Distibution Functions (BRDF's) or Bidirectional Transmittance Distribution Functions (BTDF's) to make
objects interact with light differently. This makes the objects look like they are made of different materials.

Included materials:

	- Matte
	- Glossy (Phong)
	- Reflective
	- Simple Transparency


***********************************************

	The final image is saved as
	raytrace.bmp at 1000x1000
	resolution.

***********************************************







