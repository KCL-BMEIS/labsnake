import logging

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

HIST_BINS = 100


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    fig, ax = plt.subplots(1, 1)
    line_data_r = plt.plot([])
    line_data_g = plt.plot([])
    line_data_b = plt.plot([])

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        r, bin_edges_r = np.histogram(frame[:, :, 0], bins=HIST_BINS)
        g, bin_edges_g = np.histogram(frame[:, :, 1], bins=HIST_BINS)
        b, bin_edges_b = np.histogram(frame[:, :, 2], bins=HIST_BINS)

        line_data_r.set_data(bin_edges_r, r)
        line_data_g.set_data(bin_edges_g, g)
        line_data_b.set_data(bin_edges_b, b)
        plt.pause(0.001)

        # Display the resulting frame
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q') or not cv.getWindowProperty("frame", cv.WND_PROP_VISIBLE):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
