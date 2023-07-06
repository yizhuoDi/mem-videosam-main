export CUDA_VISIBLE_DEVICES='0,1,2,3'
torchrun --nproc_per_node=4  videosam_test.py  \
  --input_image ../data/collision/ \
  --output_dir "outputs" \
  --device "cuda"
