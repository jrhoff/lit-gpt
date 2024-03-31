BASE_MODEL_DIR=$1

for file in $LORA_BASE_DIR/*.pth ; do
    
    OUT_DIR="$(echo ${file%%.*} | cut -d'-' -f1,2)"  # path with .pth removed
    echo $OUT_DIR
    # use this directory to create a litgpt checkpoint with merged weights
    python scripts/merge_lora.py --checkpoint_dir $BASE_MODEL_DIR --lora_path $file --out_dir $OUT_DIR

    # copy the config files from base model location to the merged ckpt location
    cp $BASE_MODEL_DIR/*.json $OUT_DIR/

done