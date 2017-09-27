#!/bin/zsh
#
# Copyright 2017 by Idiap Research Institute, http://www.idiap.ch
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
# Pierre-Edouard Honnet, August 2017
#
######################################################################

# Needs:
# - moses for tokenization
# - OpenNMT-py for NMT (requires CUDA, pytorch)
# - NMT models provided
# - NMT code from OpenNMT-py provided

#===== Path (to set to your system) =====#
BASEDIR=.
MOSESDIR=~/mosesdecoder
NMTDIR=~/OpenNMT-py
# Assuming you copied the model in models:
nmtmodel=$NMTDIR/models/asr2mt_model_acc_97.04_ppl_1.10_e15.pt

bash_name=$(basename $0)
function usage {
cat <<EOF
This is $bash_name
needs:
\$option
EOF
}

if [[ $# != 2 ]]
then
    usage
    exit 1
fi
input=$1
output=$2

#====== Arguments ======#
# * Input is a file containing one sentence per line
#   to be inverse-normalized
# * Output is the output file

OUTDIR=`dirname $output`
TEMPDIR=$OUTDIR/temp
bname=`basename $output`

#====== Flags ======#
# If you use Sun Grid Engine:
usesge=1
# If you want to keep temporary files, put cleaning=0:
cleaning=0

#====== Steps ======#
step1=1 # tokenization with moses
step1b=1 # get smaller length sentences if too long
step2=1 # ITN with NMT (pytorch)
#step3=0 # comparison and post-processing TODO
# Optional:
step4=0 # detokenization (not recommended)

mkdir -p $OUTDIR
mkdir -p $TEMPDIR

#====== Sequence of steps ======#
# Step 1: tokenize input with Moses
if (( $step1==1 )); then
    echo "Step 1: Tokenization"
    # Input:  $input
    # Output: $TEMPDIR/$bname.tok
    $MOSESDIR/scripts/tokenizer/tokenizer.perl -l de < $input > $TEMPDIR/$bname.tok
fi

# Step 1b: check length of sentences and cut if necessary
if (( $step1==1 )); then
    echo "Step 1b: Check sentence lengths"
    maxlen=50 # you can change it as your requirements
    # In our case we limit to 300 characters
    maxchar=`echo $maxlen | awk '{print 6*$1}'`
    # Input:  $TEMPDIR/$bname.tok
    # Output: $TEMPDIR/$bname.tok
    mv $TEMPDIR/$bname.tok $TEMPDIR/$bname.tok.tmp
    # Check if any longer than maxlen:
    while read l
    do echo $l | wc 
    done < $TEMPDIR/$bname.tok | awk '{print $2}' | sort -n | uniq | tail -1 > $TEMPDIR/count.tmp
    count=`cat $TEMPDIR/count.tmp`
    if (( $count>$maxlen )); then
	mv $TEMPDIR/$bname.tok $TEMPDIR/$bname.tok.tmp
	# Do processing to cut long ones (cutting at spaces if line
	# longer than $maxchar characters):
	fold -s -${maxchar} $TEMPDIR/$bname.tok.tmp > $TEMPDIR/$bname.tok
	rm -f $TEMPDIR/$bname.tok.tmp
	# Else keep file as it is.
    fi
    rm -f  $TEMPDIR/count.tmp
fi

# Step 2: "translate" with the NMT models
if (( $step2==1 )); then
    echo "Step 2: NMT-based ITN"
    # Input:  $TEMPDIR/$bname.tok
    # Output: $TEMPDIR/$bname.itn
    (
	cd $NMTDIR
	if (( $usesge==1 )); then
	    jobname=asr2mt-de
	    mkdir -p $TEMPDIR/gridlogs
	    logo=$TEMPDIR/gridlogs/asr2mt-de-test.o
	    loge=$TEMPDIR/gridlogs/asr2mt-de-test.e
	    # ITN stands for inverse text normalization
            qsub -l q_gpu -N $jobname -S /bin/bash -o $logo -e $loge \
		 -cwd asr2mt.job $nmtmodel $TEMPDIR/$bname.tok $TEMPDIR/$bname.itn
	else
	    # Simple case where GPU is on the same machine:
	    python $NMTDIR/translate.py -gpu 0 -model $nmtmodel \
		   -src $TEMPDIR/$bname.tok -replace_unk -verbose -output $TEMPDIR/$bname.itn
	fi
    )
fi

# Step 3: compare NMT output and input
# Not implemented yet
if (( $step3==1 )); then
    echo "Step 3: Comparing NMT-based ITN with input and post processing"
    # TODO
    # Input:  $TEMPDIR/$bname.itn
    # Output: $TEMPDIR/$bname.txt
fi

if (( $step4==1 )); then
    echo "Step 1: Detokenization"
    # Input:  $input
    # Output: $TEMPDIR/$bname.tok
    mv $output $output.tok
    $MOSESDIR/scripts/tokenizer/detokenizer.perl -l de < $output.tok > $output
fi

# Cleaning:
if (( $cleaning==1 )); then
    echo "Cleaning temporary files in: $TEMPDIR"
    rm -rf $TEMPDIR
    rm -f $output.tok
fi
