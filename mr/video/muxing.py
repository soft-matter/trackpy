# Copyright 2012 Daniel B. Allan
# dallan@pha.jhu.edu, daniel.b.allan@gmail.com
# http://pha.jhu.edu/~dallan
# http://www.danallan.com
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

import os, subprocess, re, argparse, string
import logging
from datetime import datetime, timedelta
import mr.sql
import logging
from mr.utils import extract, timestamp, time_interval

logger = logging.getLogger(__name__)

def video_info(filepath):
    """Return some video meta information as a dictionary."""
    ffmpeg = subprocess.Popen("ffmpeg -i " + filepath, shell=True, stderr=subprocess.PIPE)
    stdout, stderr = ffmpeg.communicate()
    info = {}
    info['creation'] = extract('creation_time[ ]+: ([0-9-]* [0-9:]*)', 
                               stderr, 1, timestamp)
    info['duration'] = extract('Duration: ([0-9:\.]*)', stderr, 1)
    info['detected fps'] = extract('([0-9]*.?[0-9]*) fps,', stderr, 1, float)
    info['w'], info['h'] = extract('Stream.*, ([0-9]+)x([0-9]+)', 
                                   stderr, (1,2), lambda (x,y): (int(x),int(y)))
    return info

def vls(directory='.', t0=None):
    """A video ls command.
    List meta info for the videos in a directory, including their age,
    which is computed using their creation time relative to t0."""
    if not t0:
        t0 = get_t0(directory)
    FIELD_LENGTHS = (20, 12, 20, 12, 6, 10) # 80 characters wide
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        info = video_info(filepath)
        age = info['creation'] - t0 if (t0 and info['creation']) else None
        dim = '{}x{}'.format(info['w'], info['h']) 
        fields = (filename, age, info['creation'], 
                  info['duration'], info['detected fps'], dim)
        str_fields = map(lambda (field, length): string.ljust(str(field), length),
                         zip(fields, FIELD_LENGTHS))
        line = ''.join(str_fields)
        print line

def new_directory(trial, stack, base_directory):
    """Make a directory for the muxed images. Return its path as a
    format template, which will direct the output of FFmpeg."""
    stackcode = 'T{}S{}'.format(trial, stack)
    path = os.path.join(base_directory, stackcode)
    if os.path.exists(path):
        raise ValueError, """GADS! The directory {} already exists.
                          Aborting.""".format(path)
    os.makedirs(path)
    logging.info('New directory: %s', path)
    output_template = os.path.join(path, stackcode + 'F%05d.png')
    return output_template

def _build_command(video_file, output_template, vstart, duration,
                  detected_fps, manual_fps=None, crop=None,
                  safe_mode=False):
    "Assemble the specifications into a list of command-line arguments."
    command = ['ffmpeg']
    # Putting -ss before -i is much faster, but it can fail.
    # Example: From one raw AVI I got a folder of plain gray images.
    if safe_mode:
        command.extend(['-i', video_file, '-ss', str(vstart)])
    else: 
        command.extend(['-ss', str(vstart), '-i', video_file])
    if crop:
        command.extend(['-vf', 'crop=', crop])
    assert (manual_fps or detected_fps), \
        "I need either a manual_fps or a detected_fps."
    fps = detected_fps if detected_fps else manual_fps
    command.extend(['-r', str(fps)]) 
    command.extend(['-t', str(duration)])
    command.extend(['-f', 'image2', '-pix_fmt', 'gray', output_template])
    return command

def count_files(path):
    """Count the files in a directory. Arg 'path' can be output_template.
    I call this to verify that FFmpeg actually made files."""
    directory = os.path.dirname(path)
    return len(os.listdir(directory))

def _spawn_ffmpeg(command):
    """Open an FFmpeg process, log its output, wait for it to finish,
    and return the process's return code."""
    logging.info('Command: ' + ' '.join(command))
    command = map(str, command) # to be sure
    ffmpeg = subprocess.Popen(command, shell=False, stderr=subprocess.PIPE)
    logging.info("FFmpeg is running. It began at %s.", datetime.now())
    logging.debug(ffmpeg.communicate())
    # This will wait for muxer to terminate.
    # Do not replace it with muxer.wait(). The stream from stderr can block 
    # the pipe, which deadlocks the process. Happily, communicate() relieves it.
    logging.info("FFmpeg returned at %s with the code %s.",
                 datetime.now(), ffmpeg.returncode)
    logging.info("The output directory contains %s files.",
                 count_files(command[-1]))
    return ffmpeg.returncode

def auto_name(start):
    "Name a stack based on its age, rounded to minutes."
    return '{:d}m'.format(int(round(start.seconds/60.)))

def _maybe_cast_time(time_or_string):
    if type(time_or_string) is str:
        return time_interval(time_or_string)
    else:
        return time_or_string

def mux_video(trial, video_file, vstart, duration=None, end=None, fps=None,
              crop_blackmagic=False, safe_mode=False, name=None,
              base_directory=os.path.expanduser('~/Frames')):
    vstart, duration, end = map(_maybe_cast_time, (vstart, duration, end))
    duration = duration if duration else end - vstart
    info = video_info(video_file)
    logging.info("User-specified video file %s", video_file)
    directory, filename = os.path.split(video_file)
    t0 = get_t0(directory)
    if t0 and info['creation']:
        video_age = info['creation'] - t0
        start = video_age + vstart
        end = video_age + vstart + duration
        logging.info("Ages to be muxed: %s - %s", start, end)
        if not name:
            name = auto_name(start)
    else:
        start, end = None, None
    stack = sql.new_stack(trial, name, video_file, vstart, duration, start, end)
    output_template = new_directory(trial, stack, base_directory)
    command = _build_command(video_file, output_template, vstart, duration,
                            info['detected fps'], fps, crop_blackmagic,
                            safe_mode) 
    returncode = _spawn_ffmpeg(command)
    if not returncode == 0:
        logging.error("FFmpeg returned %s.", returncode)

def which_video(directory, t0, target_start, 
                target_duration=None, target_end=None):
    """Take a directory of videos and a desired range of time (relative to t0).
    Return a video that spans that time and when to start relative to the
    timeline of the video."""
    if target_duration:
        target_end = target_start + target_duration
    table = {}
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        info = video_info(filepath)
        if not (info['creation'] and info['duration']):
            continue
        video_duration = time_interval(info['duration']) # TO DO: Do this in video_info()
        start = info['creation'] - t0
        table[filepath] = (start, start + video_duration)
    for filepath, age_range in table.iteritems():
        start, end = age_range
        if start < target_start and end > target_start:
            # This video covers the beginning of our target range.
            if end > target_end:
                # This video also covers the end.
                vstart = target_start - start
                logging.info("Matched video file: %s", filepath)
                return filepath, vstart
    return None

def mux_age(trial, start, duration=None, end=None, fps=None,
            crop_blackmagic=False, safe_mode=False, directory='.', name=None,
            base_directory=os.path.expanduser('~/Frames')):
    "Find a video covering a span of ages, and send it to mux_video."
    start, duration, end = map(time_interval, (start, duration, end))
    t0 = get_t0(directory)
    if not t0:
        raise ValueError, "I cannot slice videos by age with an age_zero file."
    video = which_video(directory, t0, start, duration, end) 
    if not video:
        logging.critical("No video in %s covers that age range.", directory)
        return False
    video_file, vstart = video
    mux_video(trial, video_file, vstart, duration, end, fps, 
              crop_blackmagic, safe_mode, name)

def set_t0(directory='.', offset=None, age_zero=None):
    """Save a plain file with age zero."
    It can be computed with reference to the first video (offset) or
    given absolutely (age_zero)."""
    if offset:
        table = {}
        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)
            creation_time, video_duration, detected_fps, w, h = \
                video_info(filepath)
            if not creation_time:
                continue
            table[filename] = creation_time
        if len(a) == 0:
            raise ValueError, """None of the videos in {} know their
                              creation time.""".format(directory)
        first_video = min(table)
        age_zero = table[first_video] - offset
        logging.info("age_zero computed with reference to first "
                     "video, %s", age_zero)
    elif age_zero:
        logging.info("age_zero given explicitly by user: %s ", age_zero)
    else:
        raise ValueError, "You must provide offset or age_zero."
    filepath = os.path.join(directory, 'age_zero')
    f = open(filepath, 'w')
    f.write(str(age_zero))
    f.close()

def get_t0(directory):
    try:
        filepath = os.path.join(directory, 'age_zero')
        f = open(filepath, 'r')
        return timestamp(f.readline())
    except IOError:
        return None
