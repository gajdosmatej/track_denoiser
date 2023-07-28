#!/bin/bash
#PBS -N batch_job
#PBS -l select=1:mem=16gb:scratch_local=1gb:ncpus=16
#PBS -l walltime=3:00:00

DATADIR=/storage/plzen1/home/gajdoma6/data/simulated/

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp /storage/praha1/home/gajdoma6/trace_bkgnd_gen.py $SCRATCHDIR
cd $SCRATCHDIR

module load python36-modules
python trace_bkgnd_gen.py -n 1e6 -t 3D -p $DATADIR

clean_scratch
