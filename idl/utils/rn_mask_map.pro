;
; Make a mask of the input data given some input
; data, a level, and a comparison type
;

FUNCTION rn_mask_map,data,level,comparison
  sz = size(data,/dim)
  answer = intarr(sz[0],sz[1])

  if comparison eq 'ge' then begin
     data_above_index = where(data ge level)
     answer(data_above_index) = 1
  endif

  if comparison eq 'le' then begin
     data_below_index = where(data le level)
     answer(data_below_index) = 1
  endif

  return,answer
END
