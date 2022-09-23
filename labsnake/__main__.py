import logging

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import soundcard as sc

logger = logging.getLogger(__name__)

HIST_BINS = 12
SAMPLE_RATE = 48000
NUM_AUDIO_SAMPLES = 1024

VIDEO_DISPLAY_ON = True
VIDEO_HISTOGRAM_ON = True
AUDIO_SCOPE_ON = True


def calc_bin_centres(bin_edges):
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def main():
    camera = cv.VideoCapture(0)
    microphone = sc.default_microphone()
    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    hist_fig = None
    if VIDEO_HISTOGRAM_ON:
        hist_fig, hist_ax = plt.subplots(1, 1)
        hist_ax.set_xlabel('Pixel intensity')
        hist_ax.set_ylabel('Frequency')
        hist_ax.set_xlim([0, 255])
        hist_ax.set_ylim([0, 1e6])

        # Init Histogram
        r_line, = hist_ax.plot(np.arange(HIST_BINS), np.zeros(HIST_BINS), 'r')
        g_line, = hist_ax.plot(np.arange(HIST_BINS), np.zeros(HIST_BINS), 'g')
        b_line, = hist_ax.plot(np.arange(HIST_BINS), np.zeros(HIST_BINS), 'b')

    audio_fig = None
    if AUDIO_SCOPE_ON:
        audio_fig, sound_ax = plt.subplots(1, 1)
        sound_ax.set_ylim([-1., 1.])
        sound_ax.set_xlabel('Time (ms)')
        sound_ax.set_ylabel('Amplitude')

        # init audio scope
        time_in_ms = 1e3 * np.linspace(0, NUM_AUDIO_SAMPLES / SAMPLE_RATE - 1 / NUM_AUDIO_SAMPLES, NUM_AUDIO_SAMPLES)
        sound_line_l, = sound_ax.plot(time_in_ms, np.zeros(NUM_AUDIO_SAMPLES))
        sound_line_r, = sound_ax.plot(time_in_ms, np.zeros(NUM_AUDIO_SAMPLES))

    # LOOP
    while True:

        if VIDEO_DISPLAY_ON or VIDEO_HISTOGRAM_ON:
            frame_read_successful, frame = camera.read()
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

            r_line.set_data(bin_centres_r, r)
            g_line.set_data(bin_centres_g, g)
            b_line.set_data(bin_centres_b, b)

        if AUDIO_SCOPE_ON:
            with microphone.recorder(samplerate=SAMPLE_RATE) as mic:
                mic_data = mic.record(numframes=NUM_AUDIO_SAMPLES)

            sound_line_l.set_ydata(mic_data[:, 0])
            # sound_line_r.set_ydata(mic_data[:, 1])

        if AUDIO_SCOPE_ON or VIDEO_HISTOGRAM_ON:
            plt.pause(1e-3)  # this redraws the MATPLOTLIB plot

        if VIDEO_DISPLAY_ON:
            # Display the video frame using OpenCV
            cv.imshow('frame', frame)

        if (
                cv.waitKey(1) == ord('q')  # detect keypress with CV window focus.
                # The wait is required to display the video frame.
                or (hist_fig and not cv.getWindowProperty("frame", cv.WND_PROP_VISIBLE))  # detect CV window close
                or (audio_fig and not plt.fignum_exists(audio_fig.number))  # detect MATPLOTLIB window close
        ):
            break

    # When everything done, release the capture
    camera.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
