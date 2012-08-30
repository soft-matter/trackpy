#!/usr/bin/python

import os, subprocess, re, argparse, string
import logging
from datetime import datetime, timedelta
from ConfigParser import ConfigParser
import MySQLdb

def extract(pattern, string, group, convert=None):
    """Extract a pattern from a string. Optionally, convert it
    to a desired type (float, timestamp, etc.) by specifying a function.
    When the pattern is not found, gracefully return None."""
    # group may be 1, (1,) or (1, 2).
    if type(group) is int:
        grp = (group,)
    elif type(group) is tuple:
        grp = group
    assert type(grp) is tuple, "The arg 'group' should be an int or a tuple."
    try:
        result = re.search(pattern, string, re.DOTALL).group(*grp)
    except AttributeError:
        # For easy unpacking, when a tuple is expected, return a tuple of Nones.
        return None if type(group) is int else (None,)*len(group)
    return convert(result) if convert else result

def timestamp(ts_string):
    "Convert a timestamp string to a datetime type."
    return datetime.strptime(ts_string, '%Y-%m-%d %H:%M:%S')

def dtparse(raw):
    m = re.match('([0-9][0-9]):([0-5][0-9]):([0-5][0-9])', raw)
    if m:
        # time only, returned as timedelta
        h, m, s = map(int, m.group(1,2,3))
        return timedelta(hours=h, minutes=m, seconds=s)
    else:
        # full datetime, returned as such
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")

def video_info(filepath):
    "Return video meta information by spawning ffmpeg and parsing its output."
    p = subprocess.Popen("ffmpeg -i " + filepath, shell=True, stderr=subprocess.PIPE)
    info = p.communicate()[1] # Read the stderr from FFmpeg.
    creation_time = extract('creation_time[ ]+: ([0-9-]* [0-9:]*)', info, 1, timestamp)
    video_duration = extract('Duration: ([0-9:\.]*)', info, 1)
    fps = extract('([0-9]*.?[0-9]*) fps,', info, 1, float)
    w, h = extract('Stream.*, ([0-9]+)x([0-9]+)', info, (1,2), 
                   lambda (x,y): (int(x),int(y)))
    return creation_time, video_duration, fps, w, h

def ls(directory, t0=None):
    "List some meta info for the videos in a directory."
    FIELD_LENGTHS = (20, 12, 20, 12, 6, 10) # 80 characters wide
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        creation_time, video_duration, fps, w, h = video_info(filepath)
        age = creation_time - t0 if (t0 and creation_time) else None
        dim = str(w)+'x'+str(h)
        fields = (filename, age, creation_time, video_duration, fps, dim)
        str_fields = map(lambda (field, length): string.ljust(str(field), length),
                         zip(fields, FIELD_LENGTHS))
        line = ''.join(str_fields)
        print line

def summary(args):
    """SUBCOMMAND: Call ls."""
    # This wrapper, while stupid, keeps ls simple and general
    # by handling the object-y command-line arguments here.
    if args.age_zero:
        ls(args.path, args.age_zero)
    else:
        ls(args.path)

def connect():
    "Return an open connection to the database."
    try:
        conn = MySQLdb.connect(host='localhost', user='scientist',
                               passwd='scientist', db='exp3')
    except MySQLdb.Error, e:
        print "Cannot connect to database."
        print "Error code:", e.args[0]
        print "Error message:", e.args[1]
        exit(1)
    return conn

def new_stack(trial, video_file, vstart, duration, start, end):
    "Insert a stack into the database, and return its id number."
    # Args start, end are relative to age_zero. If unknown or N/A,
    # do not call them, or call them as None.
    conn = connect()
    c = conn.cursor()
    c.execute("""INSERT INTO Stack (trial, video, start, end, """
              """vstart, vduration, status) VALUES """
              """(%s, %s, %s, %s, %s, %s, %s)""", 
              (trial, video_file, start, end, vstart, duration, 'reserved'))
    stack = c.lastrowid
    c.close()
    conn.close()
    logging.info('New stack: trial=%s, stack=%s' % (trial, stack))
    return stack

def new_directory(trial, stack, base_directory):
    """Make a directory for the muxed images. Return its path as a
    format template, which will direct the output of FFmpeg."""
    stackcode = 'T' + str(trial) + 'S' + str(stack) 
    path = os.path.join(base_directory, stackcode)
    assert not os.path.exists(path), \
        "GADS! The directory " + path + "already exists. Aborting."
    os.makedirs(path)
    logging.info('New directory: ' + path)
    output_template = os.path.join(path, stackcode + 'F%05d.png')
    return output_template

def build_command(video_file, output_template, vstart, duration,
                  detected_fps, manual_fps=None, crop=None):
    "Assemble the specifications into a list of command-line arguments."
    command = ['ffmpeg', '-ss', str(vstart), '-i', video_file]
    if crop:
        command.extend(['-vf', 'crop=', crop])
    assert (manual_fps or detected_fps), \
        "I need either a manual_fps or a detected_fps."
    if not manual_fps:
        fps = detected_fps
    command.extend(['-r', str(fps)]) 
    command.extend(['-t', str(duration)])
    command.extend(['-f', 'image2', '-pix_fmt', 'gray', output_template])
    return command

def spawn_ffmpeg(command):
    """Open an FFmpeg process, log its output, wait for it to finish,
    and return the process's return code."""
    logging.info('Command: ' + ' '.join(command))
    command = map(str, command) # to be sure
    muxer = subprocess.Popen(command, shell=False, stderr=subprocess.PIPE)
    logging.info("FFmpeg is running. It began at " + str(datetime.now()) + ".")
    logging.debug(muxer.communicate())
    # This will wait for muxer to terminate.
    # Do not replace it with muxer.wait(). The stream from stderr can block 
    # the pipe, which deadlocks the process. Happily, communicate() relieves it.
    logging.info("FFmpeg returned at " + str(datetime.now()) + \
                 " with the code " + str(muxer.returncode) + ".")
    return muxer.returncode

def video(args):
    "SUBCOMMAND: Mux a video, referring to a timespan in the video's timeframe."
    base_directory = args.FRAME_REPOSITORY
    trial = args.trial
    vstart = args.start
    duration = args.duration if args.duration else args.end - vstart
    for video_file in args.video_file:
        creation_time, video_duration, \
            detected_fps, w, h = video_info(video_file)
        logging.info("User-specified video file " + video_file)
        if args.age_zero:
            video_age = creation_time - args.age_zero
            start = video_age + vstart
            end = video_age + vstart + duration
            logging.info("Ages to be muxed " + str(start) + ' - ' + str(end))
        else:
            start, end = None, None
        stack = new_stack(trial, video_file, vstart, duration, start, end)
        output_template = new_directory(trial, stack, base_directory)
        command = build_command(video_file, output_template, vstart, duration,
                                detected_fps, args.fps, args.crop_blackmagic) 
        returncode = spawn_ffmpeg(command)
        assert returncode == 0, \
            "FFmpeg returned " + str(returncode) + ". See log."

def which_video(directory, t0, target_start, 
                target_end=None, target_duration=None):
    """Examine the videos in a directory and find one that covers a
    target range in age with reference to age zero, t0.
    The range can be given in terms of an end point or a duration."""
    if target_duration:
        target_end = target_start + target_duration
    duration = target_duration if target_duration else target_end - target_start
    table = {}
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        creation_time, video_duration, detected_fps, w, h = video_info(filepath)
        if not (creation_time and video_duration):
            continue
        video_duration = dtparse(video_duration) # TO DO: Do this in video_info()
        start = creation_time - t0
        table[filepath] = (start, start + video_duration)
    for filepath, age_range in table.iteritems():
        start, end = age_range
        if start < target_start and end > target_start:
            # This video covers the beginning of our target range.
            if end > target_end:
                # This video also covers the end.
                vstart = target_start - start
                creation_time, video_duration, detected_fps, \
                    w, h = video_info(filepath) # need fresh detected_fps
                return filepath, vstart, duration, detected_fps 
    return None

def age(args):
    """SUBCOMMAND: Mux a video, referring to a timespan of age 
    with reference to t0."""
    base_directory = args.FRAME_REPOSITORY
    trial = args.trial
    start = args.age
    end, duration = args.end, args.duration # One is None.
    video = which_video(args.video_directory, args.age_zero, target_start=start,
                        target_end=end, target_duration=duration)
    assert video, "No video in " + base_directory + " covers that age range."
    video_file, vstart, duration, detected_fps = video
    if not end:
        end = start + duration
    logging.info("Matched video file: " + video_file)
    logging.info("Ages to be muxed: " + str(start) + ' - ' + str(end))
    stack = new_stack(trial, video_file, vstart, duration, start, end)
    output_template = new_directory(trial, stack, base_directory)
    command = build_command(video_file, output_template, vstart, duration,
                            detected_fps, args.fps, args.crop_blackmagic) 
    returncode = spawn_ffmpeg(command)
    assert returncode == 0, \
        "FFmpeg returned " + str(returncode) + ". See log."

class ParseTime(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            values = [ self.parse(v) for v in values ]
        else:
            values = self.parse(values)
        # Save the results in the namespace using the destination
        # variable given to our constructor.
        setattr(namespace, self.dest, values)
    def parse(self, raw):
        m = re.match('([0-9][0-9]):([0-5][0-9]):([0-5][0-9])', raw)
        if m:
            # time only, returned as timedelta
            h, m, s = map(int, m.group(1,2,3))
            return timedelta(hours=h, minutes=m, seconds=s)
        else:
            # full datetime, returned as such
            return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='mux.log')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser(prog='mux')
subparsers = parser.add_subparsers()

parser_s = subparsers.add_parser('ls', help="List videos with select meta info.")
parser_s.add_argument('path', nargs='?', default='.')
parser_s.add_argument('-t0', '--age_zero', default=None, action=ParseTime)
parser_s.set_defaults(func=summary)

parser_v = subparsers.add_parser('video', help="Generate a folder of images from a specified time span of video.")
parser_v.add_argument('video_file', nargs='*')
parser_v.add_argument('-T', '--trial', required=True)
parser_v.add_argument('-ss', '-s', '--start', required=True, action=ParseTime)
group_v = parser_v.add_mutually_exclusive_group(required=True)
group_v.add_argument('-d', '--duration', action=ParseTime)
group_v.add_argument('-e', '--end', action=ParseTime)
parser_v.add_argument('-cb', '--crop_blackmagic', action='store_const', const='in_w-160-170:in_h-18-67:160:18')
parser_v.add_argument('-r', '--fps')
parser_v.add_argument('--FRAME_REPOSITORY', default=os.environ['FRAME_REPOSITORY'])
parser_v.add_argument('-t0', '--age_zero', default=None, action=ParseTime)
parser_v.add_argument('--no_sql') # TO DO: Allow user to explicitly specify directory.
parser_v.set_defaults(func=video)

parser_a = subparsers.add_parser('age', help="Generate a folder of images from a specified age range, with reference to age zero, t0.")
parser_a.add_argument('video_directory', nargs='?', default='.')
parser_a.add_argument('-T', '--trial', required=True)
parser_a.add_argument('-a', '--age', required=True, action=ParseTime)
group_a = parser_a.add_mutually_exclusive_group(required=True)
group_a.add_argument('-d', '--duration', action=ParseTime)
group_a.add_argument('-e', '--end', action=ParseTime)
parser_a.add_argument('-cb', '--crop_blackmagic', action='store_const', const='in_w-160-170:in_h-18-67:160:18')
parser_a.add_argument('-r', '--fps')
parser_a.add_argument('--FRAME_REPOSITORY', default=os.environ['FRAME_REPOSITORY'])
parser_a.add_argument('-t0', '--age_zero', default=None, action=ParseTime)
parser_a.add_argument('--no_sql') # TO DO: Allow user to explicitly specify directory.
parser_a.set_defaults(func=age)

if os.path.isfile('age_zero'):
    line = open('age_zero').readline()
    age_zero = dtparse(line[:19]) # chop off the line break, I guess
    parser_s.set_defaults(age_zero=age_zero)
    parser_v.set_defaults(age_zero=age_zero)
    parser_a.set_defaults(age_zero=age_zero)
args = parser.parse_args()
args.func(args)

