import logging

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from labsnake.timer import Timer

logger = logging.getLogger(__name__)

SCOPE_RECORD_LENGTH = 1024
SCOPE_SAMPLE_RATE = 1e3

HIST_BINS = 12

VIDEO_DISPLAY_ON = True
VIDEO_HISTOGRAM_ON = False
OSCILLOSCOPE_ON = False


def calc_bin_centres(bin_edges):
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def main():

    print("Running main")

    logging.basicConfig(level=logging.INFO)

    # Devices
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    # INIT
    hist_fig, hist_ax = plt.subplots(1, 1)
    hist_ax.set_xlabel('Pixel intensity')
    hist_ax.set_ylabel('Frequency')
    hist_ax.set_xlim([0, 255])
    hist_ax.set_ylim([0, 1e6])

    # Init Histogram
    r_line, = hist_ax.plot(np.arange(HIST_BINS), np.zeros(HIST_BINS), 'r')
    g_line, = hist_ax.plot(np.arange(HIST_BINS), np.zeros(HIST_BINS), 'g')
    b_line, = hist_ax.plot(np.arange(HIST_BINS), np.zeros(HIST_BINS), 'b')

    scope_fig, scope_ax = plt.subplots(1, 1)
    scope_ax.set_ylim([-0.1, 0.1])
    scope_ax.set_xlabel('Time (ms)')
    scope_ax.set_ylabel('Amplitude')

    # init scope
    duration = SCOPE_RECORD_LENGTH / SCOPE_SAMPLE_RATE
    dt = 1 / SCOPE_SAMPLE_RATE
    time_in_ms = 1e3 * np.linspace(0, duration - dt, SCOPE_RECORD_LENGTH)
    scope_line, = scope_ax.plot(time_in_ms, np.zeros(SCOPE_RECORD_LENGTH))

    # Timers
    camera_read_timer = Timer(label="Camera frame grabbing")
    histogram_rendering_timer = Timer(label="Histogram rendering")
    rendering_loop_timer = Timer(label="Rendering loop")
    scope_acquisition_timer = Timer(label="Scope recording timer")

    # LOOP
    while True:
        with rendering_loop_timer:

            if VIDEO_DISPLAY_ON or VIDEO_HISTOGRAM_ON:
                with camera_read_timer:
                    frame_read_successful, frame = camera.read()
                    if not frame_read_successful:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break

            if VIDEO_HISTOGRAM_ON:
                # OpenCV colour format is BGR
                b, bin_edges_b = np.histogram(frame[:, :, 0], bins=HIST_BINS)
                g, bin_edges_g = np.histogram(frame[:, :, 1], bins=HIST_BINS)
                r, bin_edges_r = np.histogram(frame[:, :, 2], bins=HIST_BINS)

                bin_centres_r = calc_bin_centres(bin_edges_r)
                bin_centres_g = calc_bin_centres(bin_edges_g)
                bin_centres_b = calc_bin_centres(bin_edges_b)

                r_line.set_data(bin_centres_r, r)
                g_line.set_data(bin_centres_g, g)
                b_line.set_data(bin_centres_b, b)

                hist_fig.canvas.draw()

                hist_figure_bitmap = np.frombuffer(hist_fig.canvas.tostring_rgb(), dtype=np.uint8)
                hist_figure_bitmap = hist_figure_bitmap.reshape(hist_fig.canvas.get_width_height(
                    physical=True)[::-1] + (3,))

                cv.imshow('hist', hist_figure_bitmap)

            # Oscilloscope
            if OSCILLOSCOPE_ON:

                with scope_acquisition_timer:
                    scope_data = np.random.randn(SCOPE_RECORD_LENGTH)
                scope_line.set_ydata(scope_data)

                scope_fig.canvas.draw()

                scope_figure_bitmap = np.frombuffer(scope_fig.canvas.tostring_rgb(), dtype=np.uint8)
                scope_figure_bitmap = scope_figure_bitmap.reshape(scope_fig.canvas.get_width_height(
                    physical=True)[::-1] + (3,))

                cv.imshow('scope', scope_figure_bitmap)

            if VIDEO_DISPLAY_ON:
                # Display the video frame using OpenCV
                cv.imshow('video_frame', frame)

            if (
                    cv.waitKey(1) == ord('q')  # detect keypress with CV window focus.
                    # The wait is required to display the video frame.
                    or not cv.getWindowProperty("frame", cv.WND_PROP_VISIBLE)  # detect CV window close
            ):
                break

    # When everything done, release the capture
    camera.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
