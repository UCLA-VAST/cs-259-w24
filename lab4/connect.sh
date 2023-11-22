sudo docker run \
    -w $PWD \
    -v /home:/home \
    -v /opt/xilinx:/opt/xilinx \
    -v /tools:/tools \
    -it ghcr.io/ucla-vast/merlin-ucla