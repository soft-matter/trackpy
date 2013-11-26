import numpy as np
import cv2

def play(frames, wait=10, label=True):
    cv2.namedWindow('playback')
    FONT = cv2.FONT_HERSHEY_SIMPLEX 
    MARGIN = 20
    try:
        for i, frame in enumerate(frames):
            try:
                frame_no = frames.cursor - 1
            except AttributeError:
                frame_no = i
            if label:
                text_pos = (20, 80)
                cv2.putText(
                    frame,"Frame %d" % frame_no, text_pos, FONT, 2, 255,
                    thickness=4, linetype=cv2.CV_AA)
            cv2.imshow('playback', frame)
            cv2.waitKey(wait) 
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow('playback')
        cv2.waitKey(1)  # a bug work-around to poke destroy
    cv2.destroyWindow('playback')
    cv2.waitKey(1)  # a bug work-around to poke destroy
