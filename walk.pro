function checkpoint, prompt_text
	rp = ''
	read,rp,prompt='Checkpoint: ' + prompt_text
	if (rp eq '') then return, 1 else return, 0
end

function check_for_existing, trial, stack, version
    ; Are there any entries in RFeature with this trial, stack, version?
    ; If so, return 1.
    rowcount = get_sql("SELECT COUNT(*) FROM RFeature WHERE trial=" + $
                         strcompress(string(trial),/remove_all) + $
                         " AND stack=" + $
                         strcompress(string(stack),/remove_all) + $
                         " AND version=" + $
                         strcompress(string(version),/remove_all))
    if (rowcount ne 0) then return, 1 else return, 0
end

pro inserttrack, tm, trial, stack, version
    ; Insert tm into the database, after checking for duplicates.
    if (check_for_existing(trial, stack, version) eq 1) then $
        message, "The RFeature table already contains data with that ID.\n" + $
                 "Choose new version number, or delete the old data."
    conn = openconn()
    s = size(tm)
    ; Insert just the data into a temp table.
    ; Then transfer it into the main table while filling in IDs.
    conn->fast_exec_sql, "DROP TABLE IF EXISTS NewRegisteredFeatures"
    conn->fast_exec_sql, "CREATE TEMPORARY TABLE NewRegisteredFeatures (" + $
                         "x float, y float, mass float, size float, " + $
                         "ecc float, frame smallint unsigned, probe int unsigned)"
    for j=0, s[2]-1 do begin
        conn->fast_exec_sql, $
            "INSERT INTO NewRegisteredFeatures " + $
            "(x, y, mass, size, ecc, frame, probe) VALUES (" + $
            strjoin(tm[0:4,j], ", ") + ", " + $
            strjoin(uint(tm[5:6,j]), ", ") + ")"
    endfor
    conn->fast_exec_sql, "INSERT INTO RFeature (trial, stack, version, " + $
                          "probe, frame, x, y, mass, size, ecc) SELECT " + $ 
                          strjoin(uint([trial, stack, version]), ", ") + ", " + $
                          "probe, frame, x, y, mass, size, ecc " + $
                          "FROM NewRegisteredFeatures"
    closeconn, conn
end

pro insertfits, fits, trial, stack, version
	conn = openconn()
	s = size(fits)
	for j=0, s[2]-1 do begin
		conn->fast_exec_sql, $
                	"INSERT INTO Probe (trial, stack, version, probe, fit_A, fit_n) VALUES (" + $
			trial + ", " + stack + ", " + version + ", " + strtrim(uint(fits[2,j]), 2) + $
			", " + strtrim(fits[0,j], 2) + ", " + strtrim(fits[1,j], 2) + ")"
	endfor
	closeconn, conn
end

pro walk, trial=trial, stack=stack, version=version, $
	goodenough=goodenough, micperpix=micperpix, timestep=timestep, skip=skip
	if not keyword_set(micperpix) then micperpix=100/427.
	if not keyword_set(goodenough) then goodenough=300
	if not (keyword_set(trial) and keyword_set(stack)) then message, 'Specify trial and stack.'
	trial = strtrim(trial, 2)
	stack = strtrim(stack, 2)
	if keyword_set(version) then version = strtrim(version, 2) else version = strtrim(1, 2)
	
	stackcode = 'T' + strtrim(trial, 2) + 'S' + strtrim(stack, 2)

	RESULTS_DIR = '~/Results/' ; Include trailing /
	path = RESULTS_DIR + stackcode + '/'
	file_mkdir, path
	message, 'Results will be saved to ' + path, /inf
	; window,0,retain=2,xsize=647,ysize=400

	pt=get_sql("SELECT x, y, mass, size, ecc, frame from UFeature where trial=" + trial + " and stack=" $
		   + stack + " and ecc < 0.1 order by frame")
	plot,pt(2,*),pt(3,*),psym=3, xtitle='mass', ytitle='size', charsize=2
	 if not checkpoint("Mass v. size for all selected probes.") then stop

	ps_start, path + stackcode + '_mass-size.ps'
	plot,pt(2,*),pt(3,*),psym=3, xtitle='mass', ytitle='size', charsize=2
	ps_end
	message, "Saved.", /inf
	t=track(pt, 8, goodenough=goodenough, mem=5)
	plot_hist,t(0,*) mod 1
	if not checkpoint("Check for pixel biasing. (This should be flat.)") then stop

	ps_start, path + stackcode + '_pixel-bias-check.ps'
	plot_hist,t(0,*) mod 1
	ps_end
	message, "Saved.", /inf
	plottr,[t[0,*]*micperpix, t[1,*]*micperpix, t[2:6,*]],g=1
	if not checkpoint("Raw trajectories.") then stop
	
	ps_start, path + stackcode + '_raw-tracks.ps', charsize=2
	plottr,[t[0,*]*micperpix, t[1,*]*micperpix, t[2:6,*]],g=1
	ps_end
	message, "Saved. Computing drift....", /inf
	mot=motion(t)
	if not checkpoint("Drift, smoothed.") then stop

	ps_start, path + stackcode + '_drift.ps'
	mot=motion(t)
	ps_end 
	message, "Saved. Subtracting drift....", /inf
	tm=rm_motion(t,mot,smooth=10)
	plottr,[[tm[0,*]*micperpix], [tm[1,*]*micperpix], tm[2:*,*]],g=1
	if not checkpoint("Trajectories, corrected for drift.") then stop

	ps_start, path + stackcode + '_tracks.ps'
	plottr,[[tm[0,*]*micperpix], [tm[1,*]*micperpix], tm[2:*,*]],g=1
	ps_end
	message, "Saved.", /inf

	if not keyword_set(skip) then begin
		frame = findfile(path + 'sample_frames/' + stackcode + 'F0*_B.png')
		a = read_png(frame[0])
		fo = fover2d(a, eclip(t,[5,1,1]), /track, /circle, rad=15)
		image_px_dim = size(a)
		; window,0,retain=2,xsize=image_px_dim[1],ysize=image_px_dim[2]
		tvscl, fo
		if not checkpoint("Annotated sample image.") then stop
	
		write_png, path + 'sample_frames/' + stackcode + '_TrackB.png', fo
		message, "Saved.", /inf
	endif

;	if checkpoint("Insert track array into RFeature table? (ENTER for yes)") then $
;		inserttrack, tm, trial, stack, version

	; exec_sql, "INSERT INTO Batch (trial, stack) VALUES (" + trial + ", " + stack + ")"
	; batch = strtrim(string(round(get_sql("SELECT @LAST_INSERT_ID"))), 2)
	batch = '6'
	print, batch
	help, batch

	; window,0,retain=2,xsize=647,ysize=400
	indmsd, tm, fits, micperpix=micperpix, timestep=timestep, export=path, trial=trial, stack=stack
	if not checkpoint("Individual probe MSDs") then stop
	
	write_text, fits, path + stackcode + '_fits.txt'
	message, "Fit results saved as text.", /inf
	
;	if checkpoint("Insert fit results into Probe table? (ENTER for yes)") then $
;		insertfits, fits, trial, stack, version

	; indmsd, tm, fits, export=path+stackcode
	if checkpoint("Proceed with ENTER if Ensemble MSD is sensible.") then begin 
		m=msd(tm, timestep=0.03333, micperpix=100/427.)
		if not checkpoint("Ensemble MSD") then stop

		write_text, m, path + stackcode + '_ensemble-msd.txt'
		message, "Saved.",/inf
	endif
end
