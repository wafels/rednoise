;
; given some data, find the level in that data such that only
; 100*top_fraction % of that data exceeds that level.
;


FUNCTION RN_FIND_LEVEL,data, top_fraction
  lev = 0
  sz = size(data,/dim)
  repeat begin
     lev = lev + 0.001*( max(data) - min(data) )
  endrep until n_elements(where(data gt lev)) lt top_fraction*double(sz[0])*double(sz[1])
  return,lev
END
