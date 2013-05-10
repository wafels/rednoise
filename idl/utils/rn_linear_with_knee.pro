;
; Higher frequencies look more like white noise. The power
; law breaks at frequency = 10^knee
;
PRO rn_linear_with_knee,x,a,f,pder
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
