#!/usr/bin/env bash

for filename in "$1"*.dot; do
    echo "$filename"
    dot "$filename" -Tpng "$filename" -o "$filename".png > /dev/null
done