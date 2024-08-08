let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd /mnt/hd/opt/comfy/jn_comfyui
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +16 nodes/meow.py
badd +847 nodes/audio.py
badd +0 /mnt/hd/opt/tts-studio/generator.py
badd +0 nodes/sampling.py
badd +40 extra/meow/tts/models/encodec.py
badd +0 nodes/image.py
badd +196 nodes/primitive_conversion.py
badd +0 /mnt/hd/opt/tts-studio/prosody.py
badd +1 __init__.py
badd +0 requirements.txt
badd +91 extra/meow/base_model.py
badd +0 /mnt/hd/opt/comfy/ComfyUI/server.py
badd +110 extra/meow/utils.py
badd +0 /mnt/hd/opt/tts-studio/auto_tune.py
badd +0 nodes/voice_fixer.py
badd +0 /mnt/hd/opt/tts-studio/environment.yml
badd +60 nodes/primitive.py
badd +1 __doc__
argglobal
%argdel
$argadd nodes/meow.py
$argadd nodes/audio.py
set stal=2
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabrewind
edit nodes/meow.py
argglobal
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 434 - ((66 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 434
normal! 0
tabnext
edit extra/meow/base_model.py
argglobal
balt nodes/meow.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 91 - ((19 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 91
normal! 02|
tabnext
edit nodes/audio.py
argglobal
2argu
balt __doc__
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 843 - ((23 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 843
normal! 0
tabnext
edit /mnt/hd/opt/tts-studio/prosody.py
argglobal
if bufexists(fnamemodify("/mnt/hd/opt/tts-studio/prosody.py", ":p")) | buffer /mnt/hd/opt/tts-studio/prosody.py | else | edit /mnt/hd/opt/tts-studio/prosody.py | endif
if &buftype ==# 'terminal'
  silent file /mnt/hd/opt/tts-studio/prosody.py
endif
balt /mnt/hd/opt/tts-studio/generator.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 69 - ((45 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 69
normal! 06|
tabnext
edit nodes/image.py
argglobal
if bufexists(fnamemodify("nodes/image.py", ":p")) | buffer nodes/image.py | else | edit nodes/image.py | endif
if &buftype ==# 'terminal'
  silent file nodes/image.py
endif
balt nodes/audio.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 194 - ((57 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 194
normal! 031|
tabnext
edit /mnt/hd/opt/tts-studio/auto_tune.py
argglobal
if bufexists(fnamemodify("/mnt/hd/opt/tts-studio/auto_tune.py", ":p")) | buffer /mnt/hd/opt/tts-studio/auto_tune.py | else | edit /mnt/hd/opt/tts-studio/auto_tune.py | endif
if &buftype ==# 'terminal'
  silent file /mnt/hd/opt/tts-studio/auto_tune.py
endif
balt /mnt/hd/opt/tts-studio/generator.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 128 - ((50 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 128
normal! 029|
tabnext
edit nodes/voice_fixer.py
argglobal
if bufexists(fnamemodify("nodes/voice_fixer.py", ":p")) | buffer nodes/voice_fixer.py | else | edit nodes/voice_fixer.py | endif
if &buftype ==# 'terminal'
  silent file nodes/voice_fixer.py
endif
balt nodes/audio.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 105 - ((25 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 105
normal! 038|
tabnext
edit extra/meow/utils.py
argglobal
if bufexists(fnamemodify("extra/meow/utils.py", ":p")) | buffer extra/meow/utils.py | else | edit extra/meow/utils.py | endif
if &buftype ==# 'terminal'
  silent file extra/meow/utils.py
endif
balt nodes/audio.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 107 - ((23 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 107
normal! 08|
tabnext
edit __init__.py
argglobal
if bufexists(fnamemodify("__init__.py", ":p")) | buffer __init__.py | else | edit __init__.py | endif
if &buftype ==# 'terminal'
  silent file __init__.py
endif
balt extra/meow/utils.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 39 - ((29 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 39
normal! 0
tabnext
edit /mnt/hd/opt/comfy/ComfyUI/server.py
argglobal
if bufexists(fnamemodify("/mnt/hd/opt/comfy/ComfyUI/server.py", ":p")) | buffer /mnt/hd/opt/comfy/ComfyUI/server.py | else | edit /mnt/hd/opt/comfy/ComfyUI/server.py | endif
if &buftype ==# 'terminal'
  silent file /mnt/hd/opt/comfy/ComfyUI/server.py
endif
balt nodes/audio.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 446 - ((32 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 446
normal! 034|
tabnext
edit __init__.py
argglobal
if bufexists(fnamemodify("__init__.py", ":p")) | buffer __init__.py | else | edit __init__.py | endif
if &buftype ==# 'terminal'
  silent file __init__.py
endif
balt /mnt/hd/opt/comfy/ComfyUI/server.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 81 - ((62 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 81
normal! 0
tabnext
edit extra/meow/base_model.py
argglobal
if bufexists(fnamemodify("extra/meow/base_model.py", ":p")) | buffer extra/meow/base_model.py | else | edit extra/meow/base_model.py | endif
if &buftype ==# 'terminal'
  silent file extra/meow/base_model.py
endif
balt nodes/audio.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 77 - ((31 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 77
normal! 017|
tabnext
edit nodes/image.py
argglobal
if bufexists(fnamemodify("nodes/image.py", ":p")) | buffer nodes/image.py | else | edit nodes/image.py | endif
if &buftype ==# 'terminal'
  silent file nodes/image.py
endif
balt nodes/audio.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1445 - ((33 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1445
normal! 010|
tabnext
edit nodes/sampling.py
argglobal
if bufexists(fnamemodify("nodes/sampling.py", ":p")) | buffer nodes/sampling.py | else | edit nodes/sampling.py | endif
if &buftype ==# 'terminal'
  silent file nodes/sampling.py
endif
balt nodes/audio.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 199 - ((0 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 199
normal! 02|
tabnext
edit /mnt/hd/opt/tts-studio/generator.py
argglobal
if bufexists(fnamemodify("/mnt/hd/opt/tts-studio/generator.py", ":p")) | buffer /mnt/hd/opt/tts-studio/generator.py | else | edit /mnt/hd/opt/tts-studio/generator.py | endif
if &buftype ==# 'terminal'
  silent file /mnt/hd/opt/tts-studio/generator.py
endif
balt nodes/audio.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1282 - ((33 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1282
normal! 033|
tabnext
edit requirements.txt
argglobal
if bufexists(fnamemodify("requirements.txt", ":p")) | buffer requirements.txt | else | edit requirements.txt | endif
if &buftype ==# 'terminal'
  silent file requirements.txt
endif
balt /mnt/hd/opt/tts-studio/generator.py
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 9 - ((8 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 9
normal! 06|
tabnext
edit /mnt/hd/opt/tts-studio/environment.yml
argglobal
if bufexists(fnamemodify("/mnt/hd/opt/tts-studio/environment.yml", ":p")) | buffer /mnt/hd/opt/tts-studio/environment.yml | else | edit /mnt/hd/opt/tts-studio/environment.yml | endif
if &buftype ==# 'terminal'
  silent file /mnt/hd/opt/tts-studio/environment.yml
endif
balt requirements.txt
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 47 - ((46 * winheight(0) + 33) / 67)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 47
normal! 07|
tabnext 4
set stal=1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
