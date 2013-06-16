import cv2

def play(frames, wait=10):
    cv2.namedWindow('playback')
    for frame in frames:
        cv2.imshow('playback', frame)
        cv2.waitKey(wait) 
