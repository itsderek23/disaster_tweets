#!/bin/sh
kaggle competitions download -p data/raw  -c nlp-getting-started
unzip -o -d data/raw data/raw/nlp-getting-started.zip
rm data/raw/nlp-getting-started.zip
