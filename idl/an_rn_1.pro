;
; Analyze some data cubes for the presence of RN
;
; Model to be fit is simple red noise
;
; Another good model to fit is a broken power law.  Higher frequencies
; look more like white noise.
;
PRO linear_with_knee,x,a,f,pder
  m = a[1]
  c = a[0]
  knee = a[2]
  f = 0.0*x
  pder = fltarr(n_elements(x),3)
  lessthan = where( x le knee, complement=morethan )
  if lessthan[0] ne -1 then begin
     ;
     ; the function itself
     ;
     f[lessthan] = m*x[lessthan] + c
     if morethan[0] ne -1 then begin
        f[morethan] = m*x[lessthan[n_elements(lessthan)-1]] + c
     endif
     ; 
     ; partial derivatives
     ;
     pder_c = 1.0 + 0.0*x

     pder_m = 0.0*x
     pder_m[lessthan]=x[lessthan]
     if morethan[0] ne -1 then begin
        pder_m[morethan]=knee
     endif

     pder_knee = 0.0*x
     if morethan[0] ne -1 then begin
        pder_knee[morethan]=m
     endif
     ;
     ; Compile them together
     ;
     pder[*,0] = pder_c[*]
     pder[*,1] = pder_m[*]
     pder[*,2] = pder_knee[*]
  endif else begin
     f[*] = m*x[0] + c
     pder_c = 1.0 + 0.0*x
     pder_m = knee + 0.0*x
     pder_knee = m + 0.0*x
     pder[*,0] = pder_c[*]
     pder[*,1] = pder_m[*]
     pder[*,2] = pder_knee[*]
  endelse
END

PRO broken_linear,x,a,f,pder
;
; Implements the log-log version of the broken power law
; End points and start points are fixed
;
  m_lower = a[1]
  c = a[0]
  breakpoint = a[2]
  m_upper = a[3]
;
  nx = n_elements(x)
  f = fltarr(nx)
  ;
  ; Portions of the spectrum before and after the break
  ;
  lessthan = where( x le breakpoint, complement=morethan )

  return
END

PRO linear_with_constant,x,a,f,pder
  m = a[1]
  c = a[0]
  const = a[2]
  f = alog( a[2] + exp(m*x+c) )
END

PRO linear,x,a,f,pder
  m = a[1]
  c = a[0]
  f = m*x+c
END


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


FUNCTION FIND_LEVEL,data, top_percentage
  lev = 0
  sz = size(data,/dim)
  repeat begin
     lev = lev + 0.001*( max(data) - min(data) )
  endrep until n_elements(where(data gt lev)) lt 0.1*double(sz[0])*double(sz[1])
  return,lev
END

PRO do_rn, root, directory, filename, eps, img_directory, function_name

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
            chisq_mask[i,j] = 1
            ;chisq_map[i,j] = chisqr[i,j]
            ;index_map[i,j] = -lf[i,j,1]
            ;tot_map[i,j] = ts_tot[i,j]
            ;max_map[i,j] = ts_max[i,j]
         endif
      endif
      
   endfor
endfor

;
; Power law index
;
power_law_raw = -reform(lf[*,*,1])
power_law_finite_index = where( finite(power_law_raw) eq 1b)
power_law_map = fltarr(nx, ny)
power_law_map(power_law_finite_index) = power_law_raw(power_law_finite_index)
;
; Mask and index of where the brightest pixels are
;
top_percentage = 0.1

tma = total(data,3,/double)
tma_level = FIND_LEVEL(tma, top_percentage)
tma_index = where(tma ge tma_level)
tma_mask = MASK_MAP(tma, tma_level, 'ge')



;
; Plots
;
charsize = 1.0

good_fits_tma_mask = chisq_mask*tma_mask
good_fits_tma_index = where(good_fits_tma_mask eq 1)

!p.multi=0
if eps eq 1 then ps,img_directory + filename+'_'+function_name + '_powerlawindex_pdf.eps', /encapsulated else window,0


h = histogram(power_law_map(good_fits_tma_index),bins=0.05, min = 0.001,loc=hloc)
h = h/total(h)
plot,hloc,h,psym=10,xtitle = 'power law index', ytitle = 'PDF',title = filename+': '+function_name,charsize=charsize
xyouts,0.0,0.1*max(h),'solid = good fit + bright',charsize=charsize
xyouts,0.0,0.2*max(h),'dotted = good fit',charsize=charsize
xyouts,0.0,0.3*max(h),'dashed = bright',charsize=charsize

h = histogram(power_law_map*chisq_mask,bins=0.05, min = 0.001,loc=hloc)
oplot,hloc,h/total(h),psym=10, linestyle=1

h = histogram(power_law_map*tma_mask,bins=0.05, min = 0.001,loc=hloc)
oplot,hloc,h/total(h),psym=10, linestyle=2

if eps eq 1 then psclose

if eps eq 1 then ps,img_directory + filename+'_'+function_name + '_image_maps.eps', /encapsulated else window,1
!p.multi=[0,3,1]
plot_image,reform(data[*,*,0])*tma_mask,title='bright emission: '+function_name,charsize=charsize

plot_image,reform(data[*,*,0])*chisq_mask,title='good fit emission: '+function_name,charsize=charsize

plot_image,reform(data[*,*,0])*good_fits_tma_mask,title='bright + good fit emission: '+function_name,charsize=charsize
!p.multi=0
if eps eq 1 then psclose



if (strpos(function_name,'knee'))[0] ne -1 then begin
   window,2
   knee_map = reform(lf[*,*,2])
   h = histogram(knee_map(good_fits_tma_index),bins=0.05,loc=hloc,max=max(x))
   plot,hloc,h/total(h),psym=10,xtitle = 'alog(knee frequency)', ytitle = 'PDF',title = filename +': bright + good fit',charsize=charsize

   h = histogram(knee_map*chisq_mask,bins=0.05,loc=hloc,max=max(x))
   oplot,hloc,h/total(h),psym=10, linestyle=1

   h = histogram(knee_map*tma_mask,bins=0.05,loc=hloc,max=max(x))
   oplot,hloc,h/total(h),psym=10, linestyle=2
endif


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

root = '/home/ireland/'
directory = 'Data/oscillations/mcateer/outgoing3/'
eps = 1
img_directory = '~/ts/img/rn/'
function_name = 'linear_with_knee'

fulllist = file_list(root + directory, files=files)
nfiles = n_elements(files)
for i = 0, nfiles-1 do begin
   filename = files[i]
   do_rn,root, directory, filename, eps, img_directory, function_name
endfor


END
