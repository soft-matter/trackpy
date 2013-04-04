import cv2
import numpy as np

def circle(features, frames):
    """Play video, circling features in each frame.

    Parameters
    ----------
    features : DataFrame including columns 'frame', 'x', and 'y'
    frames : iterable container of image arrays, like a list of images or a 
        Video object (See mr.opencv.video.Video.)
    """

    RADIUS = 10
    COLOR = (0, 200, 0)
    centers = features.set_index('frame')[['x', 'y']].apply(np.rint).astype('int')
    cv2.namedWindow("playback")
    print "Press Ctrl+C to interrupt video."
    try:
        for frame in frames: 
            # Colorize frame to allow colored annotations.
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.cv.CV_GRAY2RGB)
            frame_no = frames.cursor - 1
            these_centers = centers.loc[frame_no, ['x', 'y']]
            # This if/else statement handles the unusual case in which
            # there is only one probe in a frame.
            if isinstance(these_centers, Series):
                these_centers = list([these_centers.tolist()])
            else:
                these_centers = these_centers.values
            for x, y in these_centers:
                cv2.circle(frame, (x, y), RADIUS, COLOR) 
            cv2.imshow("playback", frame)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        return 
