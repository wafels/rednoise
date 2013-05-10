;
; Analyze some data cubes for the presence of RN
;

all_power_law = [0]
all_brightness = [0]
root = '/home/ireland/'
directory = 'Data/oscillations/mcateer/outgoing3/'
eps = 1
img_directory = '~/ts/img/rn/'
function_name = 'rn_linear_with_knee'

fulllist = file_list(root + directory, files=files)
nfiles = n_elements(files)
for i = 0, nfiles-1 do begin
   filename = files[i]
   
; get the data
   restorefull = root + directory + filename
   restore,restorefull
;
; De-rotate the data
;
  displacements = get_correl_offsets(region_window)
  datacube = image_translate(region_window, displacements, /interp)
  sz = size(datacube,/dim)
  sz[1] = sz[1] - (4+ceil(max(displacements[0])))
  datacube = datacube[0:sz[0]-1,0:sz[1]-1,0:sz[2]-1]
;
; Make a movie of the data
;

;
; Analyze the datacube and print out some results
;
   answer = RN_DO_ANALYSIS(datacube, function_name,$
                           eps=1,$
                           use_top='max',$
                           img_directory='~/ts/img/rn/',$
                           brightness_level=-1,$
                           top_fraction=0.1,$
                           img_root_name=filename)

; Keep the results from each analysis
   all_power_law = [all_power_law, answer.power_law]
   all_brightness = [all_brightness, answer.brightness]
endfor
all_power_law = all_power_law[1:*]
all_brightness = all_brightness[1:*]




END
