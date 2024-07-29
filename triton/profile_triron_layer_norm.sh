nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true --capture-range=cudaProfilerApi -o ./nsys_logs/compare_triton_layer_norm_dynamic_q python profile_triron_layer_norm.py
