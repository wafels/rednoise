;
; Read in a bunch of FITS files and save them as an IDL Save file
;

root = '/home/ireland/Data/AIA_Data/'
directory = 'SOL2011-04-30T21-45-49L061C108'
wavelength = '171'
directory = 'SOL2011-05-09T22-28-13L001C055'

save_directory = '/home/ireland/Data/rednoise/' + directory
spawn,'mkdir '+save_directory
save_directory = save_directory + '/' + wavelength
spawn,'mkdir '+save_directory
save_directory = save_directory + '/'

;
; Get a list of the FITS files
;
files = file_list(root + directory + '/' + wavelength, '*.fts')
nfiles = n_elements(files)

;
; Read in the first one to get the spatial extent of it
;
test_load = readfits(files[0])
sz = size(test_load,/dim)

;
; Define the datacube and load in the data
;
region_window = dblarr(sz[0], sz[1], nfiles)

for i = 0, nfiles -1 do begin
   data = readfits(files[i])
   region_window[*,*,i] = data[*,*]
endfor

;
;
;
save_location = save_directory + directory +'.' + wavelength + '.sav'
print,'Saving data to ' + save_location
save,filename = save_location,region_window


END
