#!/bin/bash

for pid in $(ps -ef | awk '/server.py/ {print $2}'); do kill -9 $pid; done
