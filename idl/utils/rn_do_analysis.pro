;
; Do a rednoise-type analysis on a datacube
;

FUNCTION rn_do_analysis, data, function_name,$
                         eps=eps,$
                         use_top=use_top,$
                         img_directory=img_directory,$
                         brightness_level=brightness_level,$
                         top_fraction=top_fraction,$
                         img_root_name=img_root_name 


; Print to file?
  if not(keyword_set(eps)) then eps=0

; where to save the images?
  if not(keyword_set(img_directory)) then img_directory='~/'

; which type of maximum to use
  if not(keyword_set(use_top)) then use_top='mean'

; fixed brightness level
 if not(keyword_set(brightness_level)) then brightness_level=-1

; what percentage of the emission is considered "bright"
 if not(keyword_set(top_fraction)) then top_fraction=0.05

; what percentage of the emission is considered "bright"
 if not(keyword_set(img_root_name)) then img_root_name='defaultname'

;
; which model
;
  if function_name eq 'rn_linear_with_knee' then begin
     n_parameters = 3
  endif

;
; Calculate positive non-zero frequencies
;
  sz = size(data,/dim)
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
  status_map = fltarr(nx,ny)
  lf = fltarr(nx,ny,n_parameters)
  lf_sig = fltarr(nx,ny,n_parameters)
  chisqr = fltarr(nx,ny)
  ts_av_map = fltarr(nx,ny)
  ts_max_map = fltarr(nx,ny)
  good_chisq_mask = fltarr(nx,ny)

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

        if function_name eq 'rn_linear_with_knee' then begin
         ;
         ; Get an estimate of the broken power law
         ;
           pwrlaw_estimate = linfit(x[0:n_elements(x)/5.0],y[0:n_elements(x)/5.0])
           A = [pwrlaw_estimate[0], pwrlaw_estimate[1], x[n_elements(x)/2.0]]
           weights = abs(y-poly(x,pwrlaw_estimate))^2
           yfit = CURVEFIT(x, y, 1.0/weights, A, SIGMA, FUNCTION_NAME=function_name, chisq=chisqr_fit, status = status)
           convert_to_reduced_chisqr = 1.0
        endif

        ; Store the status of the fit
        status_map[i,j] = status

        ; If the fit was good, store more information
        if status eq 0 then begin
           ; fit parameters
           lf[i,j,*] = a[*]

           ; fit uncertainty estimates
           lf_sig[i,j,*] = sigma[*]

           ; chi-squared value - note that this cannot be
           ; interpreted using the chi-squared distribution
           chisqr[i,j] = chisqr_fit/convert_to_reduced_chisqr

           ; average value of the time series
           ts_av_map[i,j] = average(ts)

           ; maximum value of the time series
           ts_max_map[i,j] = max(ts)

           ; We assume that these are the locations that have
           ; a good fit
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
  if brightness_level eq -1 then begin
     if use_top eq 'mean' then begin
        top = ts_av_map
     endif
     if use_top eq 'max' then begin
        top = ts_max_map
     endif
     top_level = RN_FIND_LEVEL(top, top_fraction)
  endif else begin
     top_level = brightness_level
  endelse
  top_index = where(top ge top_level)
  top_mask = RN_MASK_MAP(top, top_level, 'ge')
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
; Calculate a data analysis name
;
  dataname = img_root_name+'_'+function_name+'_'+use_top

;
; Plots
;
  charsize = 1.0
  !p.multi=0
;
; Histograms of power law indices
;
  if eps eq 1 then ps,img_directory + dataname+ '_powerlawindex_pdf.eps', /encapsulated else window,0

  ; Good fits, bright
  hgb = histogram(good_power_law_map(good_top_index),bins=0.05, min = 0.001,loc=hgbloc)
  hgb = hgb/total(hgb)

  ; Good fits, not bright
  hgn = histogram(good_power_law_map(good_low_index),bins=0.05, min = 0.001,loc=hgnloc)
  hgn = hgn/total(hgn)

  ; Good fits, all of them
  hga = histogram(good_power_law_map(good_index),bins=0.05, min = 0.001,loc=hgaloc)
  hga = hga/total(hga)

  yrange = minmax([hgb,hgn,hga])
  xrange = minmax([hgbloc,hgnloc,hgaloc])

  plot,hgbloc,hgb,psym=10,xtitle = 'power law index', ytitle = 'PDF',title = dataname,charsize=charsize, xrange=xrange, yrange=yrange
  xyouts,0.0,0.1*yrange[1],'solid = good fit + bright',charsize=charsize

  oplot,hgnloc,hgn,psym=10, linestyle=1
  xyouts,0.0,0.2*yrange[1],'dotted = good fit + not bright',charsize=charsize

  oplot,hgaloc,hga,psym=10, linestyle=2
  xyouts,0.0,0.3*yrange[1],'dashed = all good fits',charsize=charsize

  xyouts,0.0,0.4*yrange[1],'brightness fraction = '+trim(top_fraction)
  xyouts,0.0,0.5*yrange[1],'brightness level = '+trim(top_level)

  xyouts,0.0,0.6*yrange[1],'fraction pixels with good fits ='+trim(total(good_chisq_mask)/double(n_elements(good_chisq_mask)))

  if eps eq 1 then psclose

;
; Maps of data
;
  if eps eq 1 then ps,img_directory + dataname + '_image_maps.eps', /encapsulated,/color else window,1
  !p.multi=[0,3,2]
  loadct,3
;
; base image
;
  base_image = top
  base_range = minmax(base_image)

  ; all emission
  rn_show_image, top, base_range,'all emission',charsize

  ; bright emission
  rn_show_image, top*top_mask, base_range,'bright emission',charsize

  ; non bright emission
  rn_show_image, top*low_mask, base_range,'non bright emission',charsize

  ; good fit emission
  rn_show_image, top*good_mask, base_range,'good fit emission: '+function_name,charsize

  ; bright + good fit emission
  rn_show_image, top*good_top_mask, base_range,'bright + good fit emission: '+function_name,charsize

  ; non-bright + good fit emission
  rn_show_image, top*good_low_mask ,base_range,'non-bright + good fit emission: '+function_name,charsize

  if eps eq 1 then psclose

;
; Maps of the power law index
;
  if eps eq 1 then ps,img_directory + dataname+ '_powerlawindex_maps.eps', /encapsulated,/color else window,1
  !p.multi=[0,3,1]
  loadct,39

  base_image = good_power_law_map
  base_range = minmax(base_image)

  rn_show_image, good_power_law_map, base_range,'Power law index, good fit: '+function_name,charsize

  rn_show_image, good_power_law_map*top_mask, base_range,'Power law index, bright + good fit: '+function_name,charsize

  rn_show_image, good_power_law_map*low_mask, base_range,'Power law index, non-bright + good fit: '+function_name,charsize

  !p.multi=0
  if eps eq 1 then psclose

;
; Histogram of power law index versus brightness
;
  loadct,0

  if eps eq 1 then ps,img_directory + dataname + '_powerlawindex_brightness_histogram.eps', /encapsulated,/color else window,1
  !p.multi=0
  loadct,39

;
; Want to do a correlation between the log(brightness) and the power
; law index
;
  top_nonzero_index = where(top gt 0.0)
  top_nonzero_mask = intarr(nx,ny)
  top_nonzero_mask(top_nonzero_index) = 1

  correlation_mask = top_nonzero_mask * good_mask
  correlation_index = where(correlation_mask eq 1)

  brightness = alog10(top(correlation_index))
  power_law_index_at_brightness = power_law_raw(correlation_index)

  bin1 = 0.05
  bin2 = 0.025
  min1 = 0.001
  min2 = min(brightness)
  h2d = hist_2d(power_law_index_at_brightness, brightness, min1=min1, min2=min2, bin1=bin1, bin2=bin2)
  sz = size(h2d,/dim)
  plot_image,h2d,scale=[bin1,bin2],origin=[min1,min2],xtitle='power law index',ytitle='alog10(brightness)',/nosquare
  plots,[min1,min1+bin1*sz[0]],[alog10(top_level),alog10(top_level)],color = 255
  xyouts,min1,alog10(top_level),'brightness level, top '+trim(top_fraction),color = 255
  psclose

  loadct,0

  return,{power_law:power_law_index_at_brightness, brightness:brightness}

end



  ;; if (strpos(function_name,'knee'))[0] ne -1 then begin
  ;;    window,2
  ;;    knee_map = reform(lf[*,*,2])
  ;;    h = histogram(knee_map(good_fits_top_index),bins=0.05,loc=hloc,max=max(x))
  ;;    plot,hloc,h/total(h),psym=10,xtitle = 'alog(knee frequency)', ytitle = 'PDF',title = filename +': bright + good fit',charsize=charsize

  ;;    h = histogram(knee_map*chisq_mask,bins=0.05,loc=hloc,max=max(x))
  ;;    oplot,hloc,h/total(h),psym=10, linestyle=1

  ;;    h = histogram(knee_map*top_mask,bins=0.05,loc=hloc,max=max(x))
  ;;    oplot,hloc,h/total(h),psym=10, linestyle=2
  ;; endif

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

