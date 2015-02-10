;
; Takes level 1.0 AIA cutout service data in one directory, preps it up
; to 1.5, and dumps it in another
;

location = '~/Data/ts/request4/'
wave = '193'

indir = location + '/' + '1.0/' + wave + '/'

outdir = location + '/' + '1.5/' + wave + '/'

wavetype = '*_' + wave + '_*'


; Read in the data
print,'Reading in data from ', indir
print,'Wave type = ', wavetype
read_sdo,file_search(indir + wavetype),index,data

; Dump the data out
print,'Writing prepped data to ', outdir
print,'Wave type = ', wavetype
aia_prep,index,data,/cutout, out_dir=outdir,/do_write_fits





END
