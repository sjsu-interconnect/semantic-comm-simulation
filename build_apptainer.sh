#!/bin/bash
echo "Building Apptainer images..."
apptainer build --fakeroot sender.sif sender.def
apptainer build --fakeroot receiver.sif receiver.def
apptainer build --fakeroot channel.sif channel.def
apptainer build --fakeroot edge_encoder.sif edge_encoder.def
apptainer build --fakeroot edge_decoder.sif edge_decoder.def
echo "Done."
