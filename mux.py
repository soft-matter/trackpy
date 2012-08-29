#!/usr/bin/python

import os, subprocess, re, argparse, string
from datetime import datetime, date, time, timedelta
from dallantools import ParseTime, dtparse
from ConfigParser import ConfigParser

def extract(pattern, string, group, convert=None):
    """Extract a pattern from a string. Optionally, convert it
    to a desired type (float, timestamp, etc.) by specifying a function.
    When the pattern is not found, gracefully return None (unlike re)."""
    # group may be 1, (1,) or (1, 2).
    if type(group) is int:
        grp = (group,)
    elif type(group) is tuple:
        grp = group
    assert type(grp) is tuple, "group should be an int or a tuple."
    try:
        result = re.search(pattern, string, re.DOTALL).group(*grp)
    except AttributeError:
        # For easy unpacking, when a tuple is expected, return a tuple of Nones.
        return None if type(group) is int else (None,)*len(group)
    return convert(result) if convert else result

def timestamp(ts_string):
    "Convert a timestamp string to a datetime type."
    return datetime.strptime(ts_string, '%Y-%m-%d %H:%M:%S')

def video_info(filepath):
    "Return video meta information by spawning ffmpeg and parsing its output."
    p = subprocess.Popen("ffmpeg -i " + filepath, shell=True, stderr=subprocess.PIPE)
    info = p.communicate()[1] # Read the stderr from FFmpeg.
    creation_time = extract('creation_time[ ]+: ([0-9-]* [0-9:]*)', info, 1, timestamp)
    duration = extract('Duration: ([0-9:\.]*)', info, 1)
    fps = extract('([0-9]*.?[0-9]*) fps,', info, 1, float)
    w, h = extract('Stream.*, ([0-9]+)x([0-9]+)', info, (1,2), 
                   lambda (x,y): (int(x),int(y)))
    return creation_time, duration, fps, w, h

def ls(path, t0=None):
    "List some meta info for the videos in a directory."
    FIELD_LENGTHS = (20, 12, 20, 12, 6, 10) # 80 characters wide
    for filename in sorted(os.listdir(path)):
        filepath = os.path.join(path, filename)
        creation_time, duration, fps, w, h = video_info(filepath)
        age = creation_time - t0 if t0 else None
        dim = str(w)+'x'+str(h)
        fields = (filename, age, creation_time, duration, fps, dim)
        str_fields = map(lambda (field, length): string.ljust(str(field), length),
                         zip(fields, FIELD_LENGTHS))
        line = ''.join(str_fields)
        print line

def summary(args):
    """SUBCOMMAND: Call ls."""
    # This wrapper, while stupid, keeps ls simple and general
    # by handling the object-y command-line arguments here.
    if args.t0:
        ls(args.path, args.t0)
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

def video(args):
    "SUBCOMMAND: Mux a video, referring to a timespan in the video's timeframe."
    for video_file in args.video_file:
        info = video_info(video_file)
        if (not info):
            print 'Skipping', video_file
            continue
        trial = args.trial
        command = ['ffmpeg', '-ss', str(args.start), '-i', video_file]
        if args.crop_blackmagic:
            command.extend(['-vf', 'crop=' + args.crop_blackmagic])
        if args.fps:
            command.extend(['-r', args.fps])
        else:
            command.extend(['-r', str(info['fps'])])
        if args.duration:
            duration = args.duration 
        else:
            # Compute duration from start & end times.
            duration = end - start
        if args.t0:
            age = info['creation_time'] - args.t0
        else:
            age = None

        # Log this stack-to-be in the database, and make a directory for it.
        conn = connect('mux')
        c = conn.cursor()
        c.execute("SELECT * FROM Stack")
        if age:
            c.execute("INSERT INTO Stack (trial, video, start, end, vstart, vduration, status) VALUES (%s, %s, %s, %s, %s, %s, %s)", (trial, video_file, age + args.start, age + args.start + duration, args.start, duration, 'reserved' ))
        else:
            c.execute("INSERT INTO Stack (trial, video, vstart, vduration, status) VALUES (%s, %s, %s, %s, %s)", (trial, video_file, args.start, duration, 'reserved' ))
            c.execute("SELECT LAST_INSERT_ID()")
            stack, = c.fetchone() # type is long int
            c.close()
            stackcode = 'T' + trial + 'S' + str(stack) 
            path = os.path.join(args.FRAME_REPOSITORY, stackcode)
            output_template = os.path.join(path, stackcode + 'F%05d.png')

            if os.path.exists(path):
                print 'The path to', stackcode, 'already exists. Looks like trouble.'
                exit(1)
            else:
                os.makedirs(path)
                command.extend(['-t', str(duration), '-f', 'image2', '-pix_fmt', 'gray', output_template])

                # Start the muxing process, and update the log.
                print 'Command:', ' '.join(command)
                muxer = subprocess.Popen(map(str, command), shell=False, stderr=subprocess.PIPE)
                c = conn.cursor()
                c.execute("UPDATE Stack SET status='muxing started' WHERE trial=%s AND stack=%s", (trial, stack))
                c.execute("SELECT * FROM Stack WHERE trial=%s AND stack=%s", (trial, stack))
                print 'Row:', ' '.join(map(str, list(c.fetchone())))
                c.close()
                print 'Muxing into ' + path + '. Waiting for process termination.'
                muxer.wait()

                # The muxer is done. Update the log.
                c = conn.cursor()
                c.execute("UPDATE Stack SET status='muxed' WHERE trial=%s AND stack=%s", (trial, stack))
                c.execute("SELECT * FROM Stack WHERE trial=%s AND stack=%s", (trial, stack))
                print 'Row:', ' '.join(map(str, list(c.fetchone())))
                c.close()
                conn.close()

parser = argparse.ArgumentParser(prog='mux')
subparsers = parser.add_subparsers()

parser_s = subparsers.add_parser('ls', help='List videos with their metadata.')
parser_s.add_argument('path', nargs='?', default='.')
parser_s.add_argument('-z', '--t0', default=None, action=ParseTime)
parser_s.set_defaults(func=summary)

parser_v = subparsers.add_parser('video', help='Turn a portion of a video into an image stack, referring to a timespan in the video\'s timeframe.')
parser_v.add_argument('video_file', nargs='*')
parser_v.add_argument('-T', '--trial', required=True)
parser_v.add_argument('-ss', '-s', '--start', required=True, action=ParseTime)
group_v = parser_v.add_mutually_exclusive_group(required=True)
group_v.add_argument('-t', '-d', '--duration', action=ParseTime)
group_v.add_argument('-e', '--end', action=ParseTime)
parser_v.add_argument('-cb', '--crop_blackmagic', action='store_const', const='in_w-160-170:in_h-18-67:160:18')
parser_v.add_argument('-r', '--fps')
parser_v.add_argument('--FRAME_REPOSITORY', default=os.environ['FRAME_REPOSITORY'])
parser_v.add_argument('-z', '--t0', default=None, action=ParseTime)
parser_v.set_defaults(func=video)

parser_m = subparsers.add_parser('age', help='Specify a span of time in the timeframe of the experiment (layer age). A image stack will be created from the apporpriate video.')


if os.path.isfile('t0'):
    config = ConfigParser()
    config.read('t0')
    t0 = dtparse(config.get('Time', 't0'))
    parser_s.set_defaults(t0=t0)
    parser_v.set_defaults(t0=t0)
    #parser_v.set_defaults(fps=fps)
args = parser.parse_args()
args.func(args)

