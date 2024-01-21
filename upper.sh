#!/bin/bash
for c in 1 2 4 8 16 32 64; do
    python ~/genghan/dejavu_profile/profile.py --compress $c --record 1
done