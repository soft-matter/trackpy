Usage
=====

Choose interesting sections of video
------------------------------------

To review the inventory of videos in a folder, we have a variant of the Unix
list command, ``ls``. It shows select video meta data.

$ mux ls

To analyze a section of video, first convert it into a folder of frames. For convenience,
there are several ways to use this.
You may specify the starting and ending times::

$ mux video -s 00:01:00 -e 00:02:00 -T trial DSC0001.MOV

or the starting time and the duration::

$ mux video -s 00:01:00 -d 00:01:00 -T trial DSC0001.MOV

where trial is a number chosen by you. It is used to name the folder where the frames are output.

Sometimes, when many videos are taken in one experiment, video timecodes becoming confusing.
It is more convenient to refer to the age of the system, the time logged in your notebook.
If you tell ``mux`` the age of the first video, it will convert between age time and video time.
For example, if the first video was started 1 minute 16 seconds into the experiment, type::

$ mux set_t0 --offset 00:01:16

or in shorthand::

$ mux set_t0 -o 00:01:16

Now you can slice videos by starting age and ending age::

$ mux age -a 00:01:00 -e 00:02:00 -T trial

or by starting age and duration::

$ mux age -a 00:01:00 -d 00:01:00 -T trial

Note that we use ``-a`` in place of ``-s`` when we slice by age.

Locate probes in the frames.
----------------------------

Import the feature module in the mr package::

$ python
>>>> from mr.feature import *

List the images in a directory -- probably a directory just created by mux::

>>>> imlist = list_images('/media/Frames/T100S1/')

Try out some parameters on a sample of images from that list. Specify feature
size and minimum 'mass' (integrated brightness)::

>>>> sample(imlist, 9, minmass=1000)

Specifically, ``sample`` locates features in the first, middle, and last frame,
and displays each one in turn. Press f to enter (and exit) fullscreen mode. Close the display
window to proceed to the next one.

``sample`` is a convenience function: it wraps around a couple deeper functions, any of which
may we used individually.

>>>>
