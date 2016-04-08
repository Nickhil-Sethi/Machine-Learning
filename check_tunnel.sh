#!/bin/bash
#bash script to check if tunnel is alive

pid=`pgrep -f -x 'ssh -N -L 49999:localhost:27017 ns1635@hades0.es.its.nyu.edu'`

if [ -z "$pid" ]; then
	ssh -N -L 49999:localhost:27017 ns1635@hades0.es.its.nyu.edu
fi