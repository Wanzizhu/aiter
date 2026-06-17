docker run --rm -it --name zw_iter \
    --device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined \
    -v /home/carhuang/weizhu15/aiter:/workspace/aiter -w /workspace/aiter \
    --entrypoint /bin/bash rocm/fw-bringup:gfx1250-atom-dev-vinayak-20260606