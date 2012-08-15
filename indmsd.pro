; pro indmsd, t, minN=minN, dim=dim, micperpix=micperpix, $
; timestep=timestep, yrange=yrange
; 
; PURPOSE: Compute the mean square displacements of particles
; individually, and plot the results together.
; 
; INPUTS:
; 
; t - the output of track.pro
; 
; minN - minimum number of independent measurements. For an ENSEMBLE msd
; a value of 1000 is typical. For individual msd, we have less to work with.
; My arbitrary default is 50.
; 
; micperpix, timestep - same as msd.pro
;
; yrange - manually set yrange of plot (automated detection works poorly)
;
; OUTPUTS:
; None so far -- the results are just plotted. The time grids may be different
; for each particle, so organizing a comprehensive output array will be tricky.
; 
; MODIFICATION HISTORY:
; Designed for use with the Weeks & Crocker kit at
; http://www.physics.emory.edu/~weeks/idl/
; This code may be freely distributed if properly attributed.
; April 2011 - Written by Daniel B. Allan, Johns Hopkins University
; 

; Simple routine to set default values succinctly and inform the user.
pro set_default, arg, argname, val
	if n_elements(arg) eq 0 then begin
		arg=val
		message, argname + ' set to default.',/inf
	endif
end

pro indmsd, t, fits, cl=cl, export=export, minN=minN, dim=dim, micperpix=micperpix, timestep=timestep, yrange=yrange, $
		trial=trial, stack=stack 
	set_default, minN, 'minN',25
	set_default, dim, 'dim', 2
	set_default, micperpix, 'micperpix', 100/427.
	set_default, timestep, 'timestep', 0.03333
	set_default, yrange, 'yrange', [0.001,10]

	conn = openconn()
	help, conn
	conn->fast_exec_sql, "CREATE TEMPORARY TABLE DraftMSD (trial int unsigned, stack int unsigned, " + $
			     "probe int unsigned not null, datapoint_num int unsigned auto_increment, t float not null, x float not null, y float not null, " + $
                             "x2 float not null, y2 float not null , r2 float not null, N float not null, " + $
			     "primary key (probe, datapoint_num)) Engine=MyISAM"

	message,'Computing individual MSDs....',/inf
	; Loop over all particles in the track array, slicing
	; out one particle at a time.
	i=0
	fits = []
	if keyword_set(export) then ps_start,filename=export+'_indmsd.ps' else window,0,retain=2
	while n_elements(t(*,where(t(6,*) eq i))) gt 7 do begin
		data=msd(t(*,where(t(6,*) eq i)),minN=minN, $
		         micperpix=micperpix,timestep=timestep,/quiet)
		s = size(data)
		for j=0, s[2]-1 do begin
			conn->fast_exec_sql, "INSERT INTO DraftMSD (probe, t, x, y, x2, y2, r2, N) VALUES (" $
					    + strtrim(string(i), 2) + ", " + strjoin(data(*,j), ", ") + ")"
		endfor
		if i eq 0 then plot,data(0,*),data(2*dim+1,*),/xl,/yl,yrange=yrange,xrange=[0.01,10],xtitle="Lag time (s)",ytitle="MSD (um^2)", charsize=2 $
		else oplot,data(0,*),data(2*dim+1,*)
		fit=linfit(alog(data(0,*)),alog(data(2*dim+1,*)))
		fits=[[fits], [exp(fit[0]), fit[1], i]]
		i=i+1
	endwhile
	conn->fast_exec_sql, "UPDATE DraftMSD set trial=" + trial + ", stack=" + stack
	conn->fast_exec_sql, "INSERT INTO MSD (trial, stack, probe, datapoint_num, t, x, y, x2, y2, r2, N) SELECT " $
                              + "trial, stack, probe, datapoint_num, t, x, y, x2, y2, r2, N FROM DraftMSD" 

	closeconn, conn

	message,'Now computing ensemble MSD....',/inf
	data=msd(t,micperpix=micperpix,timestep=timestep,/quiet)
	oplot,data(0,*),data(2*dim+1,*),psym=-7, thick=3, color=200
	if keyword_set(export) then ps_end
	if keyword_set(export) then ps_start,filename=export+'_power-hist.ps' else window,1,retain=2
	plothist,fits[1,*],xrange=[0,2],bin=0.1,title="Powers Fit to Individual Probes",xtitle="Power",ytitle="Number of Probes", charsize=2
	if keyword_set(export) then ps_end
	if keyword_set(export) then ps_start,filename=export+'_fit-scatter.ps' else window,2,retain=2
	if keyword_set(cl) then begin
		cw=clust_wts([alog(fits[0,*]),fits[1,*]], N_CLUSTERS=2)
		cl=cluster([alog(fits[0,*]),fits[1,*]],cw,N_CLUSTERS=2)
		cl0 = eclip([fits,cl],[2,0,0.5])
		cl1 = eclip([fits,cl],[2,0.5,1])
		help,cl0
		help,cl1
		cgplot,cl0[1,*],cl0[0,*],/yl,psym=1,color=cgcolor('red'),xrange=[0,2],yrange=[0.01,10],xtitle="Power Fit (slope on log-log plot)",ytitle="Coefficient Fit (intercept on log-log plot)", charsize=8
		cgplot,cl1[1,*],cl1[0,*],psym=7,color=cgcolor('blue'),/overplot
	endif else begin
		plot,fits[1,*],fits[0,*],/yl,psym=1,xrange=[0,2],yrange=[0.01, 10],xtitle="Power Fit (slope on log-log plot)",ytitle="Coefficient Fit (intercept on log-log plot)", charsize=2
	endelse
	if keyword_set(export) then ps_end
end
