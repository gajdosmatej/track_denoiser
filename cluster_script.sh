#!/bin/bash
#PBS -N batch_job
#PBS -l select=1:mem=4gb:scratch_local=10gb:ngpus=1:gpu_cap=cuda60
#PBS -l walltime=0:30:00
#PBS -q gpu

DATADIR=/storage/praha1/home/gajdoma6/test

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp -r $DATADIR/data $DATADIR/denoise_traces.py $SCRATCHDIR
cd $SCRATCHDIR
mkdir $SCRATCHDIR/models

singularity exec -B $SCRATCHDIR:/scratchdir --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow:21.08-tf2-py3.SIF python /scratchdir/denoise_traces.py

cp -r $SCRATCHDIR/models/1_zx $DATADIR/output_model

clean_scratch
