#!/bin/bash


docker run \
    -it \
    --rm \
    --ipc host \
    --gpus all \
    --shm-size 14G \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -p 7860:7860 \
    -v /home/user/projects/owl/nanoowl:/nanoowl \
    nanoowl:23-01 \
    bash