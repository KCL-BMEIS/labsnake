import logging

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

HIST_BINS = 100


def calc_bin_centres(bin_edges):
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    fig, hist_ax = plt.subplots(1, 1)

    while True:
        frame_read_successful, frame = cap.read()
        if not frame_read_successful:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # OpenCV colour format is BGR
        b, bin_edges_b = np.histogram(frame[:, :, 0], bins=HIST_BINS)
        g, bin_edges_g = np.histogram(frame[:, :, 1], bins=HIST_BINS)
        r, bin_edges_r = np.histogram(frame[:, :, 2], bins=HIST_BINS)

        bin_centres_r = calc_bin_centres(bin_edges_r)
        bin_centres_g = calc_bin_centres(bin_edges_g)
        bin_centres_b = calc_bin_centres(bin_edges_b)

        # update the plot
        hist_ax.clear()
        hist_ax.plot(bin_centres_r, r, 'r')
        hist_ax.plot(bin_centres_g, g, 'g')
        hist_ax.plot(bin_centres_b, b, 'b')
        plt.pause(1e-3)  # this redraws the MATPLOTLIB plot

        # Display the video frame using OpenCV
        cv.imshow('frame', frame)

        if (
                cv.waitKey(1) == ord('q')  # detect keypress with CV window focus.
                # The wait is required to display the video frame.
                or not cv.getWindowProperty("frame", cv.WND_PROP_VISIBLE)  # detect CV window close
                or not plt.fignum_exists(fig.number)  # detect MATPLOTLIB window close
        ):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
