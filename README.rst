Usage
=====

Choose interesting sections of video
------------------------------------

To review the inventory of videos in a folder, use::

$ mux ls

It lists video files along with select meta information, like the creation time recorded by the camera.

To analyze a section of video, first convert it into a folder of frames. For convenience, you may slice it by the starting and ending times::

$ mux video -s 00:01:00 -e 00:02:00 -T trial DSC0001.MOV

or by the starting time and the duration::

$ mux video -s 00:01:00 -d 00:01:00 -T trial DSC0001.MOV

The argument ``trial`` is a number chosen by you.

Sometimes, when many videos are taken in one experiment, video timecodes becoming confusing.
It is more convenient to refer to the age of the system -- the time logged in your notebook.
If you specify the age of the first video once, ``mux`` can translate age time into video time.
For example, if the first video was started 1 minute 16 seconds into the experiment, type::

$ mux set_t0 --offset 00:01:16

Now you can slice videos by starting age and ending age::

$ mux age -a 00:15:00 -e 00:16:00 -T trial

or by starting age and duration::

$ mux age -a 00:15:00 -d 00:01:00 -T trial

``mux`` automatically find the corresponding video that spans this age, and slices the appropriate section. Note that we use ``-a`` in place of ``-s`` when we slice by age.

Locate probes.
--------------

Import the feature module in the mr package::

>>>> from mr.feature import *

List the images in a directory -- probably a directory just created by mux::

>>>> imlist = list_images('/media/Frames/T100S1/')

Try out some parameters on a sample of images from that list. Specify feature
size and minimum 'mass' (integrated brightness)::

>>>> sample(imlist, 9, minmass=1000)

Specifically, ``sample`` locates features in the first, middle, and last frames,
and it displays each one in turn. Press f to toggle fullscreen mode. Close the display
window to proceed to the next one.

``sample`` is a convenience function: it chains together several deeper functions. Sometimes
you may want to use them individually.

To get the features from a image, use ``locate``::

>>>> first_image = imlist[0]
>>>> features = locate(first_image, 9, minmass=1000)

The full list of parameters, with their default values, is::
>>>> locate (image_file, diameter, separation=None, noise_size=1, smoothing_size=None, invert=True, percentile=64, minmass=1.):

To superimpose circles on the original image, use ``annotate``::

>>>> annotate(first_image, features)

To save the output, instead of displaying it::

>>>> annotate(first_image, features, output_file='some_filename.png')

Locate all the probes.
----------------------

Having setted on good parameters using ``sample``, process the whole folder of images using ``batch``.
The features will be saved to the database, so you'll need a trial number and a stack number. If you used mux, a stack number
was automatically assigned and incorporated into the folder name.

A typical call to batch looks like::

>>>> batch(trial, stack, imlist, 9, minmass=1000)

where trial and stack are numbers, of course. ``batch`` also accepts all the optional parameters shown for ``locate`` above.

Link features into trajectories.
--------------------------------

>>>> import mr.track

Build a query that will fetch the features you found above. You can take them all::

>>>> q = query(trial, stack)

or you can filter, by a giving a SQL ``WHERE`` clause. Examples::

>>>> q = query(trial, stack, where='mass < 10000')
>>>> q = query(trial, stack, where='ecc < 0.1')
>>>> q = query(trial, stack, where='x between 100 and 200')

You can chain conditions explicitly, as in SQL, or pass them as a list for ``query`` to join::

>>>> q = query(trial, stack, where=['mass < 10000', 'ecc < 0.1', 'x between 100 and 200'])

To link the features together into trajectories, we'll pass this query and some parameters to the function ``track``. These parameters are::

>>>> track(query, max_displacement, min_appearances, memory)

Here is an example::

>>>> t=track(q, 5, 100, 3)


