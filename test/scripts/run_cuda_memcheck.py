#!/usr/bin/env python

"""This script runs cuda-memcheck on the specified unit test. Each test case
is run in its isolated process with a timeout so that:
1) different test cases won't influence each other, and
2) in case of hang, the script would still finish in a finite amount of time.
The output will be written to a log file result.log

Example usage:
    python run_cuda_memcheck.py ../test_torch.py 600

Note that running cuda-memcheck could be very slow.
"""

import asyncio
import torch
import multiprocessing
import argparse
import subprocess
import tqdm
import re
import os
import cuda_memcheck_common as cmc

ALL_TESTS = []
GPUS = torch.cuda.device_count()

# parse arguments
parser = argparse.ArgumentParser(description="Run isolated cuda-memcheck on unit tests")
parser.add_argument('filename', help="the python file for a test, such as test_torch.py")
parser.add_argument('timeout', type=int, help='kill the test if it does not terminate in a certain amount of seconds')
parser.add_argument('--strict', action='store_true',
                    help='Whether to show cublas/cudnn errors. These errors are ignored by default because'
                         'cublas/cudnn does not run error-free under cuda-memcheck, and ignoring these errors')
parser.add_argument('--nproc', type=int, default=multiprocessing.cpu_count(),
                    help='Number of processes running tests, default to number of cores in the system')
parser.add_argument('--gpus', default='all',
                    help='GPU assignments for each process, it could be "all", or : separated list like "1,2:3,4:5,6"')
args = parser.parse_args()

# Filters that ignores cublas/cudnn errors
# TODO (@zasdfgbnm): When can we remove this? Will cublas/cudnn run error-free under cuda-memcheck?
def is_ignored_only(output):
    try:
        report = cmc.parse(output)
    except cmc.ParseError:
        # in case the simple parser fails parsing the output of cuda memcheck
        # then this error is never ignored.
        return False
    count_ignored_errors = 0
    for e in report.errors:
        if 'libcublas' in ''.join(e.stack) or 'libcudnn' in ''.join(e.stack):
            count_ignored_errors += 1
    return count_ignored_errors == report.num_errors

# Set environment UNDER_CUDA_MEMCHECK=1 to allow skipping some tests
os.environ['UNDER_CUDA_MEMCHECK'] = '1'

# Discover tests:
# To get a list of tests, run:
# pytest --setup-only test/test_torch.py
# and then parse the output
proc = subprocess.Popen(['pytest', '--setup-only', args.filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = proc.communicate()
lines = stdout.decode().strip().splitlines()
for line in lines:
    if '(fixtures used:' in line:
        line = line.strip().split()[0]
        line = line[line.find('::') + 2:]
        line = line.replace('::', '.')
        ALL_TESTS.append(line)

# Run tests:
# Since running cuda-memcheck on PyTorch unit tests is very slow, these tests must be run in parallel.
# This is done by using the coroutine feature in new Python versions.  A number of coroutines are created;
# they create subprocesses and awaiting them to finish. The number of running subprocesses could be
# specified by the user and by default is the same as the number of CPUs in the machine.
# These subprocesses are balanced across different GPUs on the system by assigning one devices per process,
# or as specified by the user
progress = 0
logfile = open('result.log', 'w')
progressbar = tqdm.tqdm(total=len(ALL_TESTS))

async def run1(coroutine_id):
    global progress

    if args.gpus == 'all':
        gpuid = coroutine_id % GPUS
    else:
        gpu_assignments = args.gpus.split(':')
        assert args.nproc == len(gpu_assignments), 'Please specify GPU assignmnent for each process, separated by :'
        gpuid = gpu_assignments[coroutine_id]

    while progress < len(ALL_TESTS):
        test = ALL_TESTS[progress]
        progress += 1
        cmd = f'CUDA_VISIBLE_DEVICES={gpuid} cuda-memcheck --error-exitcode 1 python {args.filename} {test}'
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), args.timeout)
        except asyncio.TimeoutError:
            print('Timeout:', test, file=logfile)
            proc.kill()
        else:
            if proc.returncode == 0:
                print('Success:', test, file=logfile)
            else:
                stdout = stdout.decode()
                stderr = stderr.decode()
                should_display = args.strict or not is_ignored_only(stdout)
                if should_display:
                    print('Fail:', test, file=logfile)
                    print(stdout, file=logfile)
                    print(stderr, file=logfile)
                else:
                    print('Ignored:', test, file=logfile)
        del proc
        progressbar.update(1)

async def main():
    tasks = [asyncio.create_task(run1(i)) for i in range(args.nproc)]
    for t in tasks:
        await t

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
