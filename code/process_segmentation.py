import argparse
import csv
import glob
import os.path
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import runez


def merge_smad_output(csv_path):
    speech_windows_list, music_windows_list = [], []
    with open(csv_path, newline="") as csvfile:
        smad_reader = csv.reader(csvfile, delimiter="\t")

        for row in smad_reader:
            start_time_s, end_time_s, smad_label = (
                round(float(row[0]), 3),
                round(float(row[1]), 3),
                row[2],
            )
            if smad_label == "s":
                speech_windows_list.append([start_time_s, end_time_s])
            elif smad_label == "m":
                music_windows_list.append([start_time_s, end_time_s])

    merged_speech_windows_list = merge_smad_output_per_activation(speech_windows_list)
    merged_music_windows_list = merge_smad_output_per_activation(music_windows_list)
    # print("Merged speech windows {}".format(merged_speech_windows_list))
    # print("Merged music windows {}".format(merged_music_windows_list))
    return merged_speech_windows_list, merged_music_windows_list


def merge_smad_output_per_activation(
    intervals_list, min_segment_duration_ms=2000, min_silence_duration_ms=3000
):
    # min_segment_duration_ms ==> discard any segments smaller than min_segment_duration_ms
    # min_silence_duration_ms ==> if segments[1] - segments[0] < min_silence_duration merge: [aaaa] 0.3 [bbbb]

    results = []
    while len(intervals_list) > 0:
        if len(intervals_list) == 1:  # last element in the list of intervals
            results.append(intervals_list[0])
            intervals_list.pop(0)
            continue

        # if second elements begin time - first element end time < min_silence_duration_ms
        if (
            intervals_list[1][0] - intervals_list[0][1]
            < min_silence_duration_ms / 1000.0
        ):
            # set first elements begin & max of endtimes, add smad activation label {m|s}, @ intervals_list[0][02]
            tmp = [
                intervals_list[0][0],
                max(intervals_list[0][1], intervals_list[1][1]),
            ]
            intervals_list[0] = tmp
            intervals_list.pop(1)
            continue

        # else stash first element in results & remove it.
        results.append(intervals_list[0])
        intervals_list.pop(0)

    # filter out too-short-segments
    results = [
        x
        for x in results
        if (float(x[1]) - float(x[0])) >= min_segment_duration_ms / 1000.0
    ]

    return results


def read_msaf_output(csv_path):
    windows = []
    with open(csv_path, newline="") as csvfile:
        msaf_reader = csv.reader(csvfile, delimiter="\t")
        for row in msaf_reader:
            start_time_s, end_time_s = round(float(row[0]), 3), round(float(row[1]), 3)
            windows.append([start_time_s, end_time_s])
    return windows


def plot_windows(audiofilepath, speech_windows, music_windows, msaf_windows):
    audio_buffer, sr = librosa.load(audiofilepath, sr=None, mono=True)

    # buffer lengths
    len_sig = len(audio_buffer)
    smad_speech_sig = np.zeros(len_sig)
    smad_music_sig = np.zeros(len_sig)
    msaf_sig = np.zeros(len_sig)

    # setup sigs
    max_val = abs(max(audio_buffer))
    for seg in speech_windows:
        smad_speech_sig[int(seg[0] * sr) : int(seg[1] * sr)] += max_val

    for seg in music_windows:
        smad_music_sig[int(seg[0] * sr) : int(seg[1] * sr)] += max_val

    fig = plt.figure(figsize=(20, 16))
    dt = 60 * sr
    ticks = np.arange(0, len_sig + dt, dt)

    # axes 1
    axes1 = fig.add_subplot(311)
    axes1.set_autoscale_on(True)
    axes1.autoscale_view(True, True, False)
    axes1.set_ylabel("Speech", fontsize=30)
    axes1.tick_params(axis="both", which="major", labelsize=20)
    axes1.tick_params(axis="both", which="minor", labelsize=18)
    axes1.set_xticks(ticks, ["00:00", "1:00", "2:00", "3:00", "4:00", "5:00", "6:00"])
    t = np.arange(0, len_sig, 1)
    axes1.plot(
        t,
        audio_buffer[0:len_sig],
        "b-",
        t,
        smad_speech_sig[0:len_sig],
        "r-",
        linewidth=2,
    )

    # axes 2
    axes2 = fig.add_subplot(313)
    axes2.set_autoscale_on(True)
    axes2.autoscale_view(True, True, False)
    axes2.set_ylabel("Music", fontsize=30)
    axes2.tick_params(axis="both", which="major", labelsize=20)
    axes2.tick_params(axis="both", which="minor", labelsize=18)
    axes2.set_xticks(ticks, ["00:00", "1:00", "2:00", "3:00", "4:00", "5:00", "6:00"])
    axes2.plot(
        t, audio_buffer[0:len_sig], "b-", t, smad_music_sig[0:len_sig], "aquamarine"
    )

    # axes 3
    axes3 = fig.add_subplot(312)
    axes3.set_autoscale_on(True)
    axes3.autoscale_view(True, True, False)
    axes3.set_ylabel("MSAF", fontsize=30)
    axes3.tick_params(axis="both", which="major", labelsize=20)
    axes3.tick_params(axis="both", which="minor", labelsize=18)
    axes3.set_xticks(ticks, ["00:00", "1:00", "2:00", "3:00", "4:00", "5:00", "6:00"])
    axes3.plot(t, audio_buffer[0:len_sig], "b-")

    for seg in msaf_windows:
        axes3.axvspan(
            int(seg[0] * sr), int(seg[1] * sr), color=np.random.rand(3), alpha=0.5
        )

    plt.show()
    fig.tight_layout()


def snip_song_to_segments(windows, input_song_path, output_dir):
    index = 0
    for segment in windows:
        start, end = float(segment[0]), float(segment[1])
        duration = end - start
        segment_file, ext = os.path.splitext(os.path.basename(input_song_path))
        segment_file = str(index).zfill(2) + "_" + segment_file + "_" + str(start) + "_" + str(end) + ext
        segment_fullpath = os.path.join(output_dir, segment_file)
        output = runez.run(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "panic",
            "-ss",
            start,
            "-t",
            duration,
            "-i",
            "%s" % input_song_path,
            segment_fullpath,
            fatal=False,
        )
        if output is False:
            print("OOPS ffmpeg segment file failed")
        else:
            print("the segment_fullpath: " + segment_fullpath)
        index += 1


def test_main():
    audiopath = "/Users/iroro/github/djtool-crate-digging/12_Squeeze.wav"
    debug = False

    # smad
    smad_csv_path = ("/Users/iroro/github/TVSM-dataset/inference/outputs/12_Squeeze.wav.csv")
    speech_windows, music_windows = merge_smad_output(smad_csv_path)

    # msaf
    msaf_csv_path = "/Users/iroro/github/msaf/examples/output.csv"
    msaf_windows = read_msaf_output(msaf_csv_path)

    # debug
    if debug:
        print("speech_windows {}".format(speech_windows))
        print("music_windows {}".format(music_windows))
        print("msaf_windows {}".format(msaf_windows))

        # plot
        # plot_windows(audiopath, speech_windows, music_windows, msaf_windows)

    # segment song
    segment_fullpath = "/Users/iroro/github/djtool-crate-digging/test_segments/"
    snip_song_to_segments(
        windows=msaf_windows, input_song_path=audiopath, output_dir=segment_fullpath
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="input audio file dir")
    parser.add_argument(dest='input_segments_dir')
    args = parser.parse_args()

    wav_pattern = str(Path(__file__).parent.joinpath(args.input_segments_dir)) + "/*/*.csv"
    csv_file_list = glob.glob(wav_pattern, recursive=True)
    csv_file_list.sort()

    segment_fullpath = "/Users/iroro/Desktop/IO_DJTools_MusicLibrary/test_segments"
    for csv_file_path in csv_file_list:
        msaf_windows = read_msaf_output(csv_file_path)
        audiopath_bits = csv_file_path.split(".")[:-2]
        audiopath_bits.append("wav")
        audiopath = ".".join(audiopath_bits)
        print("path: {} is legit {}".format(audiopath, os.path.isfile(audiopath)))
        if os.path.isfile(audiopath):
            snip_song_to_segments(
                windows=msaf_windows, input_song_path=audiopath, output_dir=segment_fullpath
            )
