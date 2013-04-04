import cv2

def circle(features, frames):
    RADIUS = 10
    COLOR = (0, 255, 0)
    features_by_frame = features.set_index('frame')
    cv2.namedWindow("playback")
    print "Press Ctrl+C to interrupt video."
    try: 
        for frame in frames: 
            frame_no = frames.cursor - 1
            for x, y in features_by_frame.ix[frame_no, ['x', 'y']]\
                .applymap(np.rint).astype('int').values:
                cv2.circle(frame, (x, y), RADIUS, COLOR) 
            cv2.imshow("playback", frame)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        return 
