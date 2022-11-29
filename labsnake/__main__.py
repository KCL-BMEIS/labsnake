import logging

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from numpy import frombuffer
from numpy._typing import NDArray

from labsnake.timer import Timer

logger = logging.getLogger(__name__)

SCOPE_RECORD_LENGTH = 1024
SCOPE_SAMPLE_RATE = 1e3

HIST_BINS = 12

VIDEO_WINDOW_NAME = "CAMERA. Press 'q' to quit."
HISTOGRAM_WINDOW_NAME = "COLOUR HISTOGRAM"
OSCILLOSCOPE_WINDOW_NAME = "OSCILLOSCOPE"

VIDEO_DISPLAY_ON = True
VIDEO_HISTOGRAM_ON = True
OSCILLOSCOPE_ON = True


def dilate_video_frame(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv.dilate(image, cv.getStructuringElement(cv.MORPH_CROSS, (80, 80), (40, 40)))


def calc_bin_centres(bin_edges):
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def main():
    logging.basicConfig(level=logging.INFO)

    # Devices
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    buffer_size = 1024
    audio_dtype = pyaudio.paInt16
    num_audio_channels = 1
    audio_rate = 44100
    py_audio = pyaudio.PyAudio()

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
    scope_ax.set_ylim([-1, 1])
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
    video_processing_timer = Timer(label="Video processing")

    # LOOP
    stream = py_audio.open(format=audio_dtype,
                    channels=num_audio_channels,
                    rate=audio_rate,
                    input=True,
                    frames_per_buffer=buffer_size)
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

                cv.imshow(HISTOGRAM_WINDOW_NAME, hist_figure_bitmap)

            # Oscilloscope
            if OSCILLOSCOPE_ON:

                with scope_acquisition_timer:
                    scope_data = frombuffer(stream.read(buffer_size), dtype=np.int16)
                    scope_data2 = scope_data / (2 ** 15)
                    scope_line.set_ydata(scope_data2)

                scope_fig.canvas.draw()

                scope_figure_bitmap = np.frombuffer(scope_fig.canvas.tostring_rgb(), dtype=np.uint8)
                scope_figure_bitmap = scope_figure_bitmap.reshape(scope_fig.canvas.get_width_height(
                    physical=True)[::-1] + (3,))

                cv.imshow(OSCILLOSCOPE_WINDOW_NAME, scope_figure_bitmap)

            if VIDEO_DISPLAY_ON:
                # Display the video frame using OpenCV

                with video_processing_timer:
                    frame_morphed = dilate_video_frame(frame)
                cv.imshow(VIDEO_WINDOW_NAME, frame_morphed)

            if (cv.pollKey() == ord('q')
                # detect keypress with CV window focus.
                # The wait is required to display the video frame.
                or not cv.getWindowProperty(VIDEO_WINDOW_NAME, cv.WND_PROP_VISIBLE)  # detect CV window close
            ):
                break

    # When everything done, release the capture
    camera.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
