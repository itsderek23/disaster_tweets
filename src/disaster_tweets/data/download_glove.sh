#!/bin/sh
kaggle datasets download -p data/raw -d rtatman/glove-global-vectors-for-word-representation
unzip -o -d data/raw data/raw/glove-global-vectors-for-word-representation.zip
rm data/raw/glove-global-vectors-for-word-representation.zip
rm data/raw/glove.6B.50d.txt
rm data/raw/glove.6B.200d.txt
