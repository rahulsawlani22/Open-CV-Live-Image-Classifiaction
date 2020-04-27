import numpy as np
import cv2
import os

filename='video.mp4'
frames_per_seconds=24.0
my_res = '480p '

def change_res(cap,width,height):
    cap.set(3,width)
    cap.set(4,height)

std_dim={
        "480p":(640,480),
        "720p":(1280,720),
        '"1080p"':(1920,1080),
        "4k":(3840,2160)
}
video_type={
        'avi':cv2.VideoWriter_fourcc(*'XVID'),
        'mp4':cv2.VideoWriter_fourcc(*'XVID')
}

def get_video_type(filename):
    filename,ext=os.path.splitext(filename)
    if ext in video_type:
        return video_type[ext]
    return video_type['avi']

def get_dims(cap,res='1080p'):
    width,height=std_dim["480p"]
    if res in std_dim:
            width,height=std_dim[res]
    change_res(cap,width,height)
    return width,height


cap = cv2.VideoCapture(0)
dims=get_dims(cap,my_res)
print(dims)
fourcc=cv2.VideoWriter_fourcc(*'XVID')

video_type_cv2=get_video_type(filename)
out=cv2.VideoWriter(filename,fourcc,frames_per_seconds,dims)



while True:
    res,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
