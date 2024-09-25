import os

names = ['input']  #please replace your input.mp4 video

for name in names:
    os.system(f'mkdir {name}')
    os.system(f'ffmpeg -i {name}.mp4 -start_number 0 {name}/%010d.png')
