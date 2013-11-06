import cv2

def play(frames, wait=10):
    cv2.namedWindow('playback')
    try:
        for frame in frames:
            cv2.imshow('playback', frame)
            cv2.waitKey(wait) 
    except KeyboardInterrupt:
        pass
