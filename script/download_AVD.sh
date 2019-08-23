#!/bin/bash

# This is the script to download the AVD dataset
# Alternatively, you can obtain the dataset from the original sources, whom we thank for providing the data
# The relevant paper is:
# Ammirato, Phil, et al. "A dataset for developing and benchmarking active vision." ICRA, 2017.


# download file from google drive where the original AVD is stored
function gdrive_download () {
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
rm -rf /tmp/cookies.txt
}

DATADIR=../data
FILEID=0B8zhxvoYK3WTLXlvYlhrc2tDVW8 
OUTPUT=${DATADIR}/AVD_Part3.tar
mkdir -p ${DATADIR}
gdrive_download ${FILEID} ${OUTPUT}
tar -C ${DATADIR} -xvf $OUTPUT ActiveVisionDataset/Home_011_1/
rm $OUTPUT
