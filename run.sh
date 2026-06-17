
rm -f aiter/jit/module_pa*

PATH=/opt/rocm-7.13-95/bin:$PATH
LD_LIBRARY_PATH=/opt/rocm-7.13-95/lib:$LD_LIBRARY_PATH


export HIP_VISIBLE_DEVICES=2
export ENABLE_CK=0
export ENABLE_FLYDSL=0
export AITER_REBUILD=1
for i in {1..4}; do
    echo "--------------------iteration $i---------------------------"
         python op_tests/test_pa_decode_bf16_asm.py \
         -b 65 \
         -kvh 8 \
         -c 16385 8193 \
         -m 0 1 2 3 \
         --varlen \
         --scales 1.69 1.7329 1.2276 --vmask
         
         python op_tests/test_pa_decode_bf16_asm.py \
         -b 65 \
         -kvh 8 \
         -c 16385 8193 \
         -m 0 1 2 3 \
         --scales 1.69 1.7329 1.2276 --vmask         
         
         python op_tests/test_pa_decode_bf16_asm.py \
         -b 65 \
         -kvh 8 \
         -c 16385 8193 \
         -m 0 1 2 3 \
         --scales 1.69 1.7329 1.2276        


         python op_tests/test_pa_decode_bf16_asm.py \
         -b 65 \
         -kvh 8 \
         -c 16385 8193 \
         --varlen \
         -m 0 1 2 3 \
         --scales 1.69 1.7329 1.2276       
done
