#!/usr/bin/env sh
set -xe
BASE_OUTPUT_DIR="testoutput/raw"
OUTPUT_DIR_1="$BASE_OUTPUT_DIR/default"
OUTPUT_DIR_2="$BASE_OUTPUT_DIR/auto"
mkdir -p $OUTPUT_DIR_1
mkdir -p $OUTPUT_DIR_2

# Create dummy input data
for DTYPE in "uint8" "float32"; do
    echo "Test with dtype=$DTYPE"

    NAME="data_$DTYPE"
    INPUT_FILE="$BASE_OUTPUT_DIR/$NAME.vol"
    python -c "import numpy as np; np.arange(128**3, dtype=np.$DTYPE).reshape(128, 128, 128).tofile('$INPUT_FILE')"
    echo "NUM_Z = 128" > $INPUT_FILE.info
    echo "NUM_Y = 128" >> $INPUT_FILE.info
    echo "NUM_X = 128" >> ${INPUT_FILE}.info

    echo "* with --dtype, --shape and --scale"
    python -m wkcuber.convert_raw \
    --layer_name $NAME \
    --dtype $DTYPE \
    --shape 128,128,128 \
    --scale 11.24,11.24,25 \
    $INPUT_FILE $OUTPUT_DIR_1
    [ -d $OUTPUT_DIR_1/$NAME ]
    [ -d $OUTPUT_DIR_1/$NAME/1 ]
    [ $(find $OUTPUT_DIR_1/$NAME/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
    [ -e $OUTPUT_DIR_1/datasource-properties.json ]

    echo "* without --dtype, --shape and --scale"
    python -m wkcuber.convert_raw \
    --layer_name $NAME \
    $INPUT_FILE $OUTPUT_DIR_2
    [ -d $OUTPUT_DIR_2/$NAME ]
    [ -d $OUTPUT_DIR_2/$NAME/1 ]
    [ $(find $OUTPUT_DIR_2/$NAME/1 -mindepth 3 -name "*.wkw" | wc -l) -eq 1 ]
    [ -e $OUTPUT_DIR_2/datasource-properties.json ]
done