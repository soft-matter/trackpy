function find_features, image, lopass, hipass, diameter, separation, mass_cutoff
	; Do the actual work: read, band pass, find bright Gaussian peaks.
        a = 255b-read_png(image)
        b = bpass(a, lopass, hipass)
        f = feature(b, diameter, separation, masscut=mass_cutoff, /q)
	s = size(f)
	message, strtrim(s[2], 2) + ' features in ' + image, /inf
	return, f
end

function fully_manual, lopass, hipass, diameter, separation, mass_cutoff
	; Are all parameters specified?
	return, keyword_set(lopass) $
		and keyword_set(hipass) $
		and keyword_set(diameter) $
		and keyword_set(separation) $
		and keyword_set(mass_cutoff)
end

pro read_config, CONFIG_DIR, config, $
	lopass, hipass, diameter, separation, mass_cutoff
	; If any image analysis parameters are not specified in the call to locate,
	; read them from a config file. If all ARE specified, the config file need not exist.
	if not fully_manual(lopass, hipass, diameter, separation, mass_cutoff) then begin
		if not keyword_set(config) then message, "You must provide a config file or specify all parameters in the command."
		; Read an ini-formatted configation file.
       		readcol, CONFIG_DIR  + config, names, values, format='a,i', delimit='=', /silent
       		n=n_elements(names)
        	descrip=strarr(n)
	        descrip[*] = 'I' ; I for Integer
	        create_struct,config,'',names,descrip
        	FOR i=0,n-1 DO BEGIN
                	config.(i) = values[i]
        	ENDFOR
        	if not keyword_set(lopass) then lopass = config.lopass
        	if not keyword_set(hipass) then hipass = config.hipass
        	if not keyword_set(diameter) then diameter = config.diameter
        	if not keyword_set(separation) then separation = config.separation
        	if not keyword_set(mass_cutoff) then mass_cutoff = config.mass_cutoff
	endif
end

pro locate, trial=trial, stack=stack, version=version, $
	image=image, $
	batch=batch, $
	export=export, quiet=quiet, $
	config=config, lopass=lopass, hipass=hipass, diameter=diameter, separation=separation, mass_cutoff=mass_cutoff
	; This is the function that the user calls.
	; It covers several use cases, enumerated below.
	FILE_EXTENSION = 'png'
	FRAMES_REPOSITORY = '~/Frames/' ; Include trailing /
	CONFIG_DIR = '~/Results/Config/' ; Include trailing /
	RESULTS_DIR = '~/Results/' ; Include trailing /
	read_config, CONFIG_DIR, config, lopass, hipass, diameter, separation, mass_cutoff

	; Usage I: Try parameters on a specific image.
	if keyword_set(image) then begin
		message, 'Finding features in image ' + image, /inf
		f = find_features(image, lopass, hipass, diameter, separation, mass_cutoff)
        	a = 255b-read_png(image) ; Redundant, but performance isn't important here.
		image_px_dim = size(a)
        	b = bpass(a, lopass, hipass) ; Ditto.
                fo_a = fover2d(a, f, /circle, rad=15) ; fover2d has been modified to suppress tv
                fo_b = fover2d(b, f, /circle, rad=15)
		if not keyword_set(quiet) then begin
			window,0, xsize=image_px_dim[1], ysize=image_px_dim[2], retain=2
			tvscl, fo_a
			window,1, xsize=image_px_dim[1], ysize=image_px_dim[2], retain=2
			tvscl, fo_b
			window,2, retain=2, xsize=647, ysize=400
			plot_hist, f(0, *) mod 1
		endif
		if keyword_set(export) then begin
			if not (keyword_set(trial) and keyword_set(stack)) then message, 'To export, trial and stack must be specified.'
			stackcode = 'T' + strtrim(trial, 2) + 'S' + strtrim(stack, 2)
			path = RESULTS_DIR + stackcode + '/sample_frames/'
			file_mkdir, path
			message, 'Exporting annotated frames....', /inf
               		m = STREGEX(image,'(.*/)([^/]*)\.[a-zA-Z0-9]',/SUBEXPR,/EXTRACT) ; Extract filename, sans file extension (.png or whatever)
			filename = m[2]
			write_png,  path + filename + '_A.' + FILE_EXTENSION, a
			write_png, path + filename + '_B.' + FILE_EXTENSION, b
			write_png, path + filename + '_CirclesA.' + FILE_EXTENSION, fo_a
			write_png, path + filename + '_CirclesB.' + FILE_EXTENSION, fo_b
			; TODO: Export config parameters to Result/stackcode
		endif
	endif

	; Usage II: Try parameters on the first image from a stack.
	if keyword_set(trial) and keyword_set(stack) and (not keyword_set(image)) and (not keyword_set(batch)) then begin
		stackcode = 'T' + strtrim(trial, 2) + 'S' + strtrim(stack, 2)
		searchstring = FRAMES_REPOSITORY + stackcode + '/' + stackcode + 'F*.' + FILE_EXTENSION
		message, 'Search String: ' + searchstring, /inf
		images = findfile(searchstring)
		locate, image=images[0], $
			trial=trial, stack=stack, version=version, $
			export=export, quiet=quiet, $
			config=config, lopass=lopass, hipass=hipass, diameter=diameter, separation=separation, mass_cutoff=mass_cutoff
	endif

	; Usage III: Process a whole stack and insert into the database.
	if keyword_set(batch) and keyword_set(trial) and keyword_set(stack) then begin
		trial=strtrim(trial, 2)
		stack=strtrim(stack, 2)
		if keyword_set(version) then version=strtrim(version, 2) else version=strtrim(1, 2) ; Default to Version 1
		stackcode = 'T' + trial + 'S' + stack

		images = findfile(FRAMES_REPOSITORY + stackcode + '/' + stackcode + 'F*.' + FILE_EXTENSION)
		message, 'Found ' + strtrim(n_elements(images), 2) + ' images for processing.', /inf	        

		; Export sample frames, in case the user skipped this step.
		message, 'First, exporting a sample frame....', /inf
		locate, image=images[0], $
			trial=trial, stack=stack, version=version, $
			/export, /quiet, $
			config=config, lopass=lopass, hipass=hipass, diameter=diameter, separation=separation, mass_cutoff=mass_cutoff

		message, 'Opening a connection to the database....', /inf
		conn = openconn()
		message, 'Processing main loop....', /inf
        	for i=0, n_elements(images)-1 do begin
               		frame = STREGEX(images[i],'.*F([0-9]*)\.',/SUBEXPR,/EXTRACT) ; Extract frame # from filename.
               		frame = frame[1]
	               	f = find_features(images[i], lopass, hipass, diameter, separation, mass_cutoff)
			s = size(f)
                	for j=0, s[2]-1 do begin
                        	conn->fast_exec_sql, $
                                	"INSERT INTO UFeature (trial, stack, version, frame, x, y, mass, size, ecc) VALUES (" + $
                                	trial + ", " + stack + ", " + version + ", " + frame + ", " + $
                                	strjoin(f(*,j), ", ") + ")"
			endfor
                endfor
		closeconn, conn
        endif
end
