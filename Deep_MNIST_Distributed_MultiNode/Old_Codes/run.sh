#!/bin/bash
time python dist_setup.py --job_name "ps" --task_index 0 &
time python dist_setup.py --job_name "worker" --task_index 0 &
