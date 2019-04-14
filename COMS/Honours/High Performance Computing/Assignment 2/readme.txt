_____ _    _ _____             _____                      _       _   _
/ ____| |  | |  __ \   /\      / ____|                    | |     | | (_)
| |    | |  | | |  | | /  \    | |     ___  _ ____   _____ | |_   _| |_ _  ___  _ __
| |    | |  | | |  | |/ /\ \   | |    / _ \| '_ \ \ / / _ \| | | | | __| |/ _ \| '_ \
| |____| |__| | |__| / ____ \  | |___| (_) | | | \ V / (_) | | |_| | |_| | (_) | | | |
\_____|\____/|_____/_/    \_\  \_____\___/|_| |_|\_/ \___/|_|\__,_|\__|_|\___/|_| |_|



By Tamlin Love (1438243)
13/04/2019

This program runs 5 versions of image convolution in an inout image using either an averaging filter, a sharpening filter or a Sobel filter of size 3
The approaches are as follows:
	1. CPU approach - serial
	2. GPU approach - naively making a thread for each pixel and using global memory
	3. GPU approach - loading tiles of an image into shared memory
	4. GPU approach - loading the filter into constant memory
	5. GPU approach - loading the image into texture memory

==Compilation==
Using the Makefile provided, type "make" in the terminal to compile convolution.cu

NOTE: If you wish to change the filter size or the size of the tile used in the shared memory approach, you need to manually edit the following lines:
	#define CONST_FILTERSIZE ???
	#define TILE_WIDTH ???
replacing ??? with the desired sizes. Note also that the Sobel filter will not work with any value of CONST_FILTERSIZE that isn't 3.


==Usage==
A simple running of the program can be achieved by typing "./convolution" in the terminal. This will run all five approaches of convolution on the "lena.pgm" image located in the 
data directory (which should be located in the same directory as convolution.cu) using an averaging filter.

To run on a different image or using a different filter, follow these steps:

1. Add image to data directory
	Some example images are provided, including "lena.pgm", "shrek.pgm", "cage.pgm", etc.
2. Compile using make
3. Type "./convolution <<image>> <<filter>>"
	where <<image>> is the image name (e.g. "shrek.pgm") and <<filter>> is a single digit representing the filter type, 
		0 - averaging filter
		1 - sharpening filter
		2 - Sobel (edge detection) filter
4. Voila! The program has run and, if all went well, should save the result of serial convolution in the data directory and should display the time and accuracy of the various
GPU approaches.