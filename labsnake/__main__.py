import logging

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import soundcard as sc

logger = logging.getLogger(__name__)

HIST_BINS = 12


def calc_bin_centres(bin_edges):
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def main():
    camera = cv.VideoCapture(0)
    microphone = sc.default_microphone()
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    fig, (hist_ax, sound_ax) = plt.subplots(2, 1)
    hist_ax.set_xlabel('Pixel intensity')
    hist_ax.set_ylabel('Frequency')
    hist_ax.set_xlim([0, 255])
    hist_ax.set_ylim([0, 1e6])

    init = True

    with microphone.recorder(samplerate=48000) as mic:
        while True:
            frame_read_successful, frame = camera.read()
            if not frame_read_successful:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            mic_data = mic.record(numframes=1024)

            # OpenCV colour format is BGR
            b, bin_edges_b = np.histogram(frame[:, :, 0], bins=HIST_BINS)
            g, bin_edges_g = np.histogram(frame[:, :, 1], bins=HIST_BINS)
            r, bin_edges_r = np.histogram(frame[:, :, 2], bins=HIST_BINS)

            bin_centres_r = calc_bin_centres(bin_edges_r)
            bin_centres_g = calc_bin_centres(bin_edges_g)
            bin_centres_b = calc_bin_centres(bin_edges_b)

            if init:
                r_line, = hist_ax.plot(bin_centres_r, r, 'r')
                g_line, = hist_ax.plot(bin_centres_g, g, 'g')
                b_line, = hist_ax.plot(bin_centres_b, b, 'b')
                sound_line_l, = sound_ax.plot(mic_data[:,0])
                sound_line_r, = sound_ax.plot(mic_data[:,1])
                init = False
            else:
                r_line.set_data(bin_centres_r, r)
                g_line.set_data(bin_centres_g, g)
                b_line.set_data(bin_centres_b, b)
                sound_line_l.set_ydata(mic_data[:,0])
                sound_line_r.set_ydata(mic_data[:,1])

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
    camera.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
