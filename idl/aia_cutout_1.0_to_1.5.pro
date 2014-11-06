;
; Takes level 1.0 AIA cutout service data in one directory, preps it up
; to 1.5, and dumps it in another
;
pro aia_cutout_1_to_1point5,location,wave

	;location = '~/Data/AIA/shutdownfun6_6hr/disk5'
	;location = '~/Data/AIA/study2/equatorial'
	;wave = '193'

	indir = location + '/' + '1.0/' + wave + '/'
	outdir = location + '/' + '1.5/' + wave + '/'
	wavetype = '*_' + wave + '_*'

	print, 'Looking for wave = ', wave
	print, 'Looking in location = ', indir
	print, 'Level 1.5 FITS files will be saved to  = ', outdir

	; Read in the data
	read_sdo,file_search(indir + wavetype),index,data

	; Dump the data out
	aia_prep,index,data,/cutout, outdir=outdir,/do_write_fits

return



END
