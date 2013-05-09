;
; Analyze some data cubes for the presence of RN
;
; Model to be fit is simple red noise


FUNCTION MASK_MAP,data,level,comparison
  sz = size(data,/dim)
  answer = intarr(sz[0],sz[1])
  if comparison eq 'ge' then begin
     for i = 0,sz[0]-1 do begin
        for j = 0,sz[1]-1 do begin
           if data[i,j] ge level then begin
              answer[i,j] = 1
           endif
        endfor
     endfor
  endif

  if comparison eq 'le' then begin
     for i = 0,sz[0]-1 do begin
        for j = 0,sz[1]-1 do begin
           if data[i,j] le level then begin
              answer[i,j] = 1
           endif
        endfor
     endfor
  endif

  return,answer
END


FUNCTION FIND_LEVEL,data, top_fraction
  lev = 0
  sz = size(data,/dim)
  repeat begin
     lev = lev + 0.001*( max(data) - min(data) )
  endrep until n_elements(where(data gt lev)) lt top_fraction*double(sz[0])*double(sz[1])
  return,lev
END


PRO show_image,im,base_range,title, charsize
  im[0,0:1] = base_range[0:1]
  plot_image,im,title=title,charsize=charsize
  return
END




;
;
;
PRO do_rn, root, directory, filename, eps, img_directory, function_name, final_good_power_law_index, final_good_brightness

; get the data
  restorefull = root + directory + filename
  restore,restorefull

;
; which model
;
  if function_name eq 'linear_with_knee' then begin
     n_parameters = 3
  endif
  if function_name eq 'linear_with_constant' then begin
     n_parameters = 3
  endif
  if function_name eq 'simple_linear' then begin
     n_parameters = 2
  endif


;
; De-rotate the data
;
  displacements = get_correl_offsets(region_window)
  data = image_translate(region_window, displacements, /interp)

;
; Calculate positive non-zero frequencies
;
  sz = size(data,/dim)
  sz[1] = sz[1] - (4+ceil(max(displacements[0])))
  n = sz[2]
  nx = sz[0]
  ny = sz[1]
  dt = 12.0
  X = (FINDGEN((N - 1)/2) + 1)
  freq = [0.0, X, N/2, -N/2 + X]/(N*DT)
  posnonzero = where(freq > 0.0)
  pfreq = freq(posnonzero)

;
; storage arrays
;
  lf = fltarr(nx,ny,n_parameters)
  lf_sig = fltarr(nx,ny,n_parameters)
  chisqr = fltarr(nx,ny)
  ts_tot = fltarr(nx,ny)
  ts_av = fltarr(nx,ny)
  ts_max = fltarr(nx,ny)
  index_map = fltarr(nx,ny)
  chisq_map = fltarr(nx,ny)
  chisq_mask = fltarr(nx,ny)
  tot_map = fltarr(nx,ny)
  max_map = fltarr(nx,ny)
  status_map = fltarr(nx,ny)

;
; Go through each pixel and do a fit
;
  for i = 0,nx-1 do begin
     for j = 0,ny-1 do begin
        ts = reform(data[i,j,*])
        ppower = ((abs(fft(ts)))^2)[posnonzero]
        x = alog(pfreq)
        y = alog(ppower)

        if function_name eq 'simple_linear' then begin
           a = linfit(x,y,$
                      sigma=sigma,$
                      chisqr=chisqr_fit)
           convert_to_reduced_chisqr = (n_elements(ppower)-2)
           status = 0
        endif

        if function_name eq 'linear_with_knee' then begin
         ;
         ; Get an estimate of the broken power law
         ;
           pwrlaw_estimate = linfit(x[0:n_elements(x)/5.0],y[0:n_elements(x)/5.0])
           A = [pwrlaw_estimate[0], pwrlaw_estimate[1], x[n_elements(x)/2.0]]
           weights = abs(y-poly(x,pwrlaw_estimate))^2
           yfit = CURVEFIT(x, y, 1.0/weights, A, SIGMA, FUNCTION_NAME=function_name, chisq=chisqr_fit, status = status)
           convert_to_reduced_chisqr = 1.0
        endif

        status_map[i,j] = status

        if status eq 0 then begin
           lf[i,j,*] = a[*]
           lf_sig[i,j,*] = sigma[*]
           chisqr[i,j] = chisqr_fit/convert_to_reduced_chisqr
           ts_tot[i,j] = total(ts)
           ts_av[i,j] = average(ts)
           ts_max[i,j] = max(ts)

           if (chisqr[i,j] le 1.2) and (chisqr[i,j] ge 0.8) then begin
              good_chisq_mask[i,j] = 1
           endif
        endif
      
     endfor
  endfor
;
; Find where the finite power law indices are
;
  power_law_raw = -reform(lf[*,*,1])
  power_law_finite_index = where( finite(power_law_raw) eq 1b)
  finite_power_law_mask = intarr(nx,ny)
  finite_power_law_mask(power_law_finite_index) = 1
;
; Where the good fits are.  Use this mask and index to determine
; where the data that needs to analyzed is in the maps
;
  good_mask = good_chisq_mask * finite_power_law_mask
  good_index = where(good_mask eq 1)
;
; Find where the good power power law indices are
;
  good_power_law_map = fltarr(nx,ny)
  good_power_law_map(good_index) = power_law_raw(good_index)

;
; Mask and index of where the brightest pixels are
;
  top_fraction = 0.05

  top = total(data,3,/double)
  top_level = FIND_LEVEL(top, top_fraction)
  top_index = where(top ge top_level)
  top_mask = MASK_MAP(top, top_level, 'ge')
;
; Mask and index of the non-brightest pixels
; 
  low_mask = 1-top_mask
  low_index = where(low_mask eq 1)
;
; Where the bright good fits 
;
  good_top_mask = good_mask*top_mask
  good_top_index = where(good_top_mask eq 1b)
;
; Where the non-bright good fits are
;
  good_low_mask = good_mask*low_mask
  good_low_index = where(good_low_mask eq 1b)


;
; Plots
;
  charsize = 1.0
  !p.multi=0
;
; Histograms of power law indices
;
  if eps eq 1 then ps,img_directory + filename+'_'+function_name + '_powerlawindex_pdf.eps', /encapsulated else window,0

  ; Good fits, bright
  hgb = histogram(power_law_map(good_fits_top_index),bins=0.05, min = 0.001,loc=hgbloc)
  hgb = hgb/total(hgb)

  ; Good fits, not bright
  hgn = histogram(power_law_map*good_fits_low_mask,bins=0.05, min = 0.001,loc=hgnloc)
  hgn = hgn/total(hgn)

  ; Good fits, all of them
  hga = histogram(power_law_map*chisq_mask,bins=0.05, min = 0.001,loc=hgaloc)
  hga = hga/total(hga)

  yrange = minmax([hgb,hgn,hga])
  xrange = minmax([hgbloc,hgnloc,hgaloc])

  plot,hgbloc,hgb,psym=10,xtitle = 'power law index', ytitle = 'PDF',title = filename+': '+function_name,charsize=charsize, xrange=xrange, yrange=yrange
  xyouts,0.0,0.1*yrange[1],'solid = good fit + bright',charsize=charsize

  oplot,hgnloc,hgn,psym=10, linestyle=1
  xyouts,0.0,0.2*yrange[1],'dotted = good fit + not bright',charsize=charsize

  oplot,hgaloc,hga,psym=10, linestyle=2
  xyouts,0.0,0.3*yrange[1],'dashed = all good fits',charsize=charsize

  xyouts,0.0,0.4*yrange[1],'brightness fraction = '+trim(top_fraction)
  xyouts,0.0,0.5*yrange[1],'brightness level = '+trim(top_level)

  xyouts,0.0,0.6*yrange[1],'fraction pixels with good fits ='+trim(total(chisq_mask)/double(n_elements(chisq_mask)))

  if eps eq 1 then psclose

;
; Maps of data
;
  if eps eq 1 then ps,img_directory + filename+'_'+function_name + '_image_maps.eps', /encapsulated,/color else window,1
  !p.multi=[0,3,2]
  loadct,3
;
; base image
;
  base_image = top
  base_range = minmax(base_image)

  ; all emission
  show_image,top,base_range,'all emission',charsize

  ; bright emission
  im = top*top_mask
  show_image,im,base_range,'bright emission',charsize

  ; non bright emission
  im = top*low_mask  
  show_image,im,base_range,'non bright emission',charsize

  ; good fit emission
  im = top*chisq_mask
  show_image,im,base_range,'good fit emission: '+function_name,charsize

  ; bright + good fit emission
  im = top*good_fits_top_mask
  show_image,im,base_range,'bright + good fit emission: '+function_name,charsize

  ; non-bright + good fit emission
  im = top*good_fits_low_mask
  show_image,im,base_range,'non-bright + good fit emission: '+function_name,charsize

  if eps eq 1 then psclose

;
; Maps of the power law index
;
  if eps eq 1 then ps,img_directory + filename+'_'+function_name + '_powerlawindex_maps.eps', /encapsulated,/color else window,1
  !p.multi=[0,3,1]
  loadct,39

  base_image = power_law_map*chisq_mask
  base_range = minmax(base_image)

  im = power_law_map*chisq_mask
  show_image,im,base_range,'Power law index, good fit: '+function_name,charsize

  im = power_law_map*good_fits_top_mask
  show_image,im,base_range,'Power law index, bright + good fit: '+function_name,charsize

  im = power_law_map*good_fits_low_mask
  show_image,im,base_range,'Power law index, non-bright + good fit: '+function_name,charsize

  !p.multi=0
  if eps eq 1 then psclose

  loadct,0

  if (strpos(function_name,'knee'))[0] ne -1 then begin
     window,2
     knee_map = reform(lf[*,*,2])
     h = histogram(knee_map(good_fits_top_index),bins=0.05,loc=hloc,max=max(x))
     plot,hloc,h/total(h),psym=10,xtitle = 'alog(knee frequency)', ytitle = 'PDF',title = filename +': bright + good fit',charsize=charsize

     h = histogram(knee_map*chisq_mask,bins=0.05,loc=hloc,max=max(x))
     oplot,hloc,h/total(h),psym=10, linestyle=1

     h = histogram(knee_map*top_mask,bins=0.05,loc=hloc,max=max(x))
     oplot,hloc,h/total(h),psym=10, linestyle=2
  endif

  if eps eq 1 then ps,img_directory + filename+'_'+function_name + '_powerlawindex_brightness_histogram.eps', /encapsulated,/color else window,1
  !p.multi=0
  loadct,39

  brightness = top*chisq_mask
  brightness_index = where(brightness gt 0.0)
  brightness_mask = intarr(nx,ny)
  brightness_mask(brightness_index) = 1
  brightness = brightness(brightness_index)
  brightness = alog10(brightness)

  final_good_brightness = brightness
  final_good_power_law_index_index = where(chisq_mask*brightness_mask eq 1)
  final_good_power_law_index = power_law_map(final_good_power_law_index_index)

  bin1 = 0.05
  bin2 = 0.025
  min1 = 0.001
  min2 = min(brightness)
  h2d = hist_2d(power_law_map*chisq_mask, brightness, min1=min1, min2=min2, bin1=bin1, bin2=bin2)
  sz = size(h2d,/dim)
  plot_image,h2d,scale=[bin1,bin2],origin=[min1,min2],xtitle='power law index',ytitle='alog10(brightness)',/nosquare
  plots,[min1,min1+bin1*sz[0]],[alog10(top_level),alog10(top_level)],color = 255
  xyouts,min1,alog10(top_level),'brightness level, top '+trim(top_fraction),color = 255
  psclose

  loadct,0


;; window,0
;; plot,chisqr,power_law_index,psym=3,xtitle='reduced chi-squared',ytitle = 'index', title = filename +': all pixels',charsize=charsize

;; window,1
;; plot_image,index_map,xtitle='x position', ytitle='y position', title = filename +': Location of "good" fits',charsize=charsize

;; window,2
;; h = histogram(index_map,bins=0.025, min = 0.001,loc=hloc)
;; plot,hloc,h,psym=10,xtitle = 'power law index', ytitle = 'PDF',title = filename +': Good fit indices distribution',charsize=charsize

;; window,3
;; plot,index_map, tot_map, psym= 3,ytitle = 'total emission (DN)', xtitle = 'good fit index', title = filename +': Total emission per good index',charsize=charsize

;; window,4
;; h2 = histogram(index_brightest, bins=0.025, min = 0.001, max= 3.0, loc=h2loc)
;; plot,h2loc,h2,psym=10,xtitle = 'power law index', ytitle = 'PDF',title = filename +': Good fit indices for pixels with total emission in top 10%',charsize=charsize

return
end

all_power_law = [0]
all_brightness = [0]
root = '/home/ireland/'
directory = 'Data/oscillations/mcateer/outgoing3/'
eps = 1
img_directory = '~/ts/img/rn/'
function_name = 'linear_with_knee'

fulllist = file_list(root + directory, files=files)
nfiles = n_elements(files)
for i = 0, nfiles-1 do begin
   filename = files[i]
   do_rn,root, directory, filename, eps, img_directory, function_name, power_law, brightness
   all_power_law = [all_power_law, power_law]
   all_brightness = [all_brightness, brightness]
endfor
all_power_law = all_power_law[1:*]
all_brightness = all_brightness[1:*]




END
