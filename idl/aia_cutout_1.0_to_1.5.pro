;
; Takes level 1.0 AIA cutout service data in one directory, preps it up
; to 1.5, and dumps it in another
;

location = '~/Data/AIA/shutdownfun6_6hr/disk5'
wave = '211'

indir = location + '/' + '1.0/' + wave + '/'

outdir = location + '/' + '1.5/' + wave + '/'

wavetype = '*_' + wave + '_*'


; Read in the data
read_sdo,file_search(indir + wavetype),index,data

; Dump the data out
aia_prep,index,data,/cutout, out_dir=outdir,/do_write_fits





END
