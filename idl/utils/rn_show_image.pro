;
; Show an image, and force it to be scaled in the range given
; by base_range
;

PRO rn_show_image,im,base_range,title, charsize
  im[0,0:1] = base_range[0:1]
  plot_image,im,title=title,charsize=charsize
  return
END
