#!/usr/bin/python

import os, subprocess, re, argparse, string
from datetime import datetime, date, time, timedelta
from dallantools import ParseTime, dtparse, connect
from ConfigParser import ConfigParser

def video_info(filepath):
	"""Parse trial and video number from filename.
	Get video meta information by opening ffmpeg and parsing output."""
	if not (re.match('.*T([0-9]*)V([0-9]*).*', filepath) and os.path.isfile(filepath)):
		return None
	trial, recording_number = re.match('.*T([0-9]*)V([0-9]*).*', filepath).group(1,2)
	p = subprocess.Popen("ffmpeg -i " + filepath, shell=True, stderr=subprocess.PIPE)
	rawinfo = p.communicate()[1] # Read the stderr from FFmpeg.
	duration = re.match('.*Duration: ([0-9:\.]*).*', rawinfo, re.DOTALL).group(1)
	creation_time = datetime.strptime(re.match('.*creation_time   : ([0-9-]* [0-9:]*).*', rawinfo, re.DOTALL).group(1), "%Y-%m-%d %H:%M:%S")
	fps = float(re.match('.* ([0-9]*.?[0-9]*) fps,.*', rawinfo, re.DOTALL).group(1))
	return {'duration': duration, 'creation_time': creation_time, 'fps': fps, 'trial': trial, 'recording_number': recording_number, 'rawinfo': rawinfo}

def summary(args):
	"""SUBCOMMAND: List some video_info for the videos in a directory."""
	print 'V  Creation            Duration    Age'
	for filename in sorted(os.listdir(args.path)):
		info = video_info(os.path.join(args.path, filename))
		if (not info): continue
		if args.zeromark:
			age = info['creation_time'] - args.zeromark
		else:
			age = None
		print string.zfill(info['recording_number'],2), info['creation_time'], info['duration'], age

def video(args):
	"""SUBCOMMAND: Mux a video, referring to a timespan in the video's timeframe."""
	for video_file in args.video_file:
		info = video_info(video_file)
		if (not info):
			print 'Skipping', video_file
			continue
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
		if args.zeromark:
			age = info['creation_time'] - args.zeromark
		else:
			age = None

		# Log this stack-to-be in the database, and make a directory for it.
		conn = connect('mux')
		c = conn.cursor()
		c.execute("""SELECT * FROM Stack""")
		if age:
			c.execute("""INSERT INTO Stack (trial, video, start, end, vstart, vduration, status) VALUES (%s, %s, %s, %s, %s, %s, %s)""", (info['trial'], info['recording_number'], age + args.start, age + args.start + duration, args.start, duration, 'reserved' ))
		else:
			c.execute("""INSERT INTO Stack (trial, video, vstart, vduration, status) VALUES (%s, %s, %s, %s, %s)""", (info['trial'], info['recording_number'], args.start, duration, 'reserved' ))
		c.execute("""SELECT LAST_INSERT_ID()""")
		stack, = c.fetchone() # type is long int
		c.close()
		stackcode = 'T' + info['trial'] + 'S' + str(stack) 
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
		c.execute("""UPDATE Stack SET status='muxing started' WHERE trial=%s AND stack=%s""", (info['trial'], stack))
		c.execute("""SELECT * FROM Stack WHERE trial=%s AND stack=%s""", (info['trial'], stack))
		print 'Row:', ' '.join(map(str, list(c.fetchone())))
		c.close()
		print 'Muxing into ' + path + '. Waiting for process termination.'
		muxer.wait()

		# The muxer is done. Update the log.
		c = conn.cursor()
		c.execute("""UPDATE Stack SET status='muxed' WHERE trial=%s AND stack=%s""", (info['trial'], stack))
		c.execute("""SELECT * FROM Stack WHERE trial=%s AND stack=%s""", (info['trial'], stack))
		print 'Row:', ' '.join(map(str, list(c.fetchone())))
		c.close()
		conn.close()

parser = argparse.ArgumentParser(prog='mux')
subparsers = parser.add_subparsers()

parser_s = subparsers.add_parser('summary', help='Summarize timecodes of videos in a directory.')
parser_s.add_argument('path', nargs='?', default='.')
parser_s.add_argument('-z', '--zeromark', default=None, action=ParseTime)
parser_s.set_defaults(func=summary)

parser_v = subparsers.add_parser('video', help='Turn a portion of a video into an image stack, referring to a timespan in the video\'s timeframe.')
parser_v.add_argument('video_file', nargs='*')
parser_v.add_argument('-ss', '-s', '--start', required=True, action=ParseTime)
group_v = parser_v.add_mutually_exclusive_group(required=True)
group_v.add_argument('-t', '-d', '--duration', action=ParseTime)
group_v.add_argument('-e', '--end', action=ParseTime)
parser_v.add_argument('-cb', '--crop_blackmagic', action='store_const', const='in_w-160-170:in_h-18-67:160:18')
parser_v.add_argument('-r', '--fps')
parser_v.add_argument('--FRAME_REPOSITORY', default=os.environ['FRAME_REPOSITORY'])
parser_v.add_argument('-z', '--zeromark', default=None, action=ParseTime)
parser_v.set_defaults(func=video)

parser_m = subparsers.add_parser('age', help='Specify a span of time in the timeframe of the experiment (layer age). A image stack will be created from the apporpriate video.')


if os.path.isfile('zeromark'):
	config = ConfigParser()
	config.read('zeromark')
	zeromark = dtparse(config.get('Time', 'zeromark'))
	parser_s.set_defaults(zeromark=zeromark)
	parser_v.set_defaults(zeromark=zeromark)
	#parser_v.set_defaults(fps=fps)
args = parser.parse_args()
args.func(args)

