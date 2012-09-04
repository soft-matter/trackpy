1. Choose interesting sections of videos.
=====

To see your inventory, list the videos in a folder::
 $ mux ls

Generate a folder of frames from a section of video. 
Specify the starting and ending times::
 $ mux video -s 00:01:00 -e 00:02:00 -T trial DSC0001.MOV

or the starting time and the duration::
 $ mux video -s 00:01:00 -d 00:01:00 -T trial DSC0001.MOV

where trial is a number chosen by you.

For a static system, slicing videos as above is the simplest approach. 
For an aging system, it's easier to refer to the age of the experiment, the time recorded
in your lab notebook. Conveniently, mux converts between age time and the timecode of
any video. It only needs one reference: when did the first video start?
For example, if the first video was started 1 minute 16 seconds into the experiment, type::
 $ mux set_t0 --offset 00:01:16
or in shorthand::
 $ mux set_t0 -o 00:01:16

Now you can slice videos by starting age and ending age::
 $ mux age -a 00:01:00 -e 00:02:00a -T trial
or by starting age and duration::
 $ mux age -a 00:01:00 -d 00:01:00 -T trial

2. Find probes.
==

Import the feature module in the mr package::
 >>>> from mr.feature import *

List the images in a directory -- probably a directory just created by mux::
 >>>> imlist = list_images('/media/Frames/T100S1/')

Try out some parameters on a sample of images from that list. Specify feature
size and minimum 'mass' (integrated brightness)::
 >>>> sample(imlist, 9, minmass=1000)


