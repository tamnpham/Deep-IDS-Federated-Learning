#!/bin/bash
declare -i CLIENTS=4
for ((i = 0; i < $CLIENTS; i++)); do
    python client.py $i &
done
