#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Read and print test results statistics
from xml.dom import minidom
from glob import glob
import bz2
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import boto3
import datetime
import requests

class TestCase:
    def __init__(self, dom):
        self.class_name = str(dom.attributes['classname'].value)
        self.name = str(dom.attributes['name'].value)
        self.time = float(dom.attributes['time'].value)
        self.errored = len(dom.getElementsByTagName('error')) > 0
        self.failed = len(dom.getElementsByTagName('failure')) > 0
        self.skipped = len(dom.getElementsByTagName('skipped')) > 0


class TestSuite:
    def __init__(self, name):
        self.name = name
        self.test_cases = []
        self.failed_count = 0
        self.skipped_count = 0
        self.errored_count = 0
        self.total_time = 0.0

    def __repr__(self):
        rc = f'{self.name} run_time: {self.total_time:.2f} tests: {len(self.test_cases)}'
        if self.skipped_count > 0:
            rc += f' skipped: {self.skipped_count}'
        return f'TestSuite({rc})'

    def append(self, test_case):
        self.test_cases.append(test_case)
        self.total_time += test_case.time
        self.failed_count += 1 if test_case.failed else 0
        self.skipped_count += 1 if test_case.skipped else 0
        self.errored_count += 1 if test_case.errored else 0

    def print_report(self, num_longest=3):
        sorted_tests = sorted(self.test_cases, key=lambda x: x.time)
        test_count = len(sorted_tests)
        print(f"class {self.name}:")
        print(f"    tests: {test_count} failed: {self.failed_count} skipped: {self.skipped_count} errored: {self.errored_count}")
        print(f"    run_time: {self.total_time:.2f} seconds")
        print(f"    avg_time: {self.total_time/test_count:.2f} seconds")
        if test_count >= 2:
            print(f"    median_time: {statistics.median(x.time for x in sorted_tests):.2f} seconds")
        sorted_tests = sorted_tests[-num_longest:]
        print(f"    {len(sorted_tests)} longest tests:")
        for test in reversed(sorted_tests):
            print(f"        {test.name} time: {test.time:.2f} seconds")
        print("")



def parse_report(path):
    dom = minidom.parse(path)
    for test_case in dom.getElementsByTagName('testcase'):
        yield TestCase(test_case)

def parse_reports(folder):
    reports = glob(os.path.join(folder, '**', '*.xml'), recursive=True)
    tests_by_class = dict()
    for report in reports:
        for test_case in parse_report(report):
            class_name = test_case.class_name
            if class_name not in tests_by_class:
                tests_by_class[class_name] = TestSuite(class_name)
            tests_by_class[class_name].append(test_case)
    return tests_by_class

def build_info():
    return {
        "build_pr": os.environ.get("CIRCLE_PR_NUMBER"),
        "build_tag": os.environ.get("CIRCLE_TAG"),
        "build_sha1": os.environ.get("CIRCLE_SHA1"),
        "build_branch": os.environ.get("CIRCLE_BRANCH"),
        "build_job": os.environ.get("CIRCLE_JOB"),
        "build_workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID"),
    }

def build_message(test_case):
    return {
        "normal": {
            **build_info(),
            "test_suite_name": test_case.class_name,
            "test_case_name": test_case.name,
        },
        "int": {
            "time": int(time.time()),
            "test_total_count": 1,
            "test_total_time": int(test_case.time * 1000),
            "test_failed_count": 1 if test_case.failed > 0 else 0,
            "test_skipped_count": 1 if test_case.skipped > 0 else 0,
            "test_errored_count": 1 if test_case.errored > 0 else 0,
        },
    }

def send_report_to_scribe(reports):
    access_token = os.environ.get("SCRIBE_GRAPHQL_ACCESS_TOKEN")

    if not access_token:
        print("No scribe access token provided, skip sending report!")
        return
    print("Scribe access token provided, sending report...")
    url = "https://graph.facebook.com/scribe_logs"
    r = requests.post(
        url,
        data={
            "access_token": access_token,
            "logs": json.dumps(
                [
                    {
                        "category": "perfpipe_pytorch_test_times",
                        "message": json.dumps(build_message(test_case)),
                        "line_escape": False,
                    }
                    for name in sorted(reports.keys())
                    for test_case in reports[name].test_cases
                ]
            ),
        },
    )
    r.raise_for_status()

def assemble_s3_object(reports, *, total_seconds):
    return {
        **build_info(),
        'total_seconds': total_seconds,
        'suites': {
            name: {
                'total_seconds': suite.total_time,
                'cases': [
                    {
                        'name': case.name,
                        'seconds': case.time,
                        'errored': case.errored,
                        'failed': case.failed,
                        'skipped': case.skipped,
                    }
                    for case in suite.test_cases
                ],
            }
            for name, suite in reports.items()
        }
    }

def send_report_to_s3(obj):
    job = os.environ.get('CIRCLE_JOB')
    sha1 = os.environ.get('CIRCLE_SHA1')
    branch = os.environ.get('CIRCLE_BRANCH', '')
    if branch not in ['master', 'nightly'] and not branch.startswith("release/"):
        print("S3 upload only enabled on master, nightly and release branches.")
        print(f"skipping test report on branch: {branch}")
        return
    now = datetime.datetime.utcnow().isoformat()
    key = f'test_time/{sha1}/{job}/{now}Z.json.bz2'  # Z meaning UTC
    s3 = boto3.resource('s3')
    try:
        s3.get_bucket_acl(Bucket='ossci-metrics')
    except Exception as e:
        print(f"AWS ACL failed: {e}")
    print("AWS credential found, uploading to S3...")

    obj = s3.Object('ossci-metrics', key)
    print("")
    # use bz2 because the results are smaller than gzip, and the
    # compression time penalty we pay is only about half a second for
    # input files of a few megabytes in size like these JSON files, and
    # because for some reason zlib doesn't seem to play nice with the
    # gunzip command whereas Python's bz2 does work with bzip2
    obj.put(Body=bz2.compress(json.dumps(obj).encode()))

def print_suite_regression_info(name, info):
    print()
    print(f"class {name}:")
    missing = info.get("missing")
    if missing:
        print("    missing in current run,")

def print_regressions(obj, *, n, stdev_threshold):
    sha1 = os.environ.get("CIRCLE_SHA1")

    print()

    print("Following output is to check this commit for test time regressions:")
    print(f"    {sha1}")

    base = subprocess.check_output(
        ["git", "merge-base", sha1, "origin/master"],
        encoding="ascii",
    ).strip()

    # if current commit is already on master, we need to exclude it from
    # this history; otherwise we include the merge-base
    commits = subprocess.check_output(
        ["git", "rev-list", f"--max-count={n+1}", base],
        encoding="ascii",
    ).splitlines()
    if base == sha1:
        commits = commits[1:]
    else:
        commits = commits[:-1]

    job = os.environ.get("CIRCLE_JOB")
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(name="ossci-metrics")
    index = {}
    for commit in commits:
        summaries = bucket.objects.filter(Prefix=f"test_time/{commit}/{job}/")
        index[commit] = list(summaries)

    objects = {}
    # should we do these in parallel?
    for commit, summaries in index.items():
        objects[commit] = []
        for summary in summaries:
            binary = summary.get()['Body'].read()
            string = bz2.decompress(binary).decode('utf-8')
            objects[commit].append(json.loads(string))

    print()

    print(f"Comparing test times for job {job} against base commit and its {n-1} most recent ancestors:")
    for commit in commits:
        runs = objects[commit]
        num_runs = len(runs)
        prefix = str(num_runs).rjust(3)
        plural = " " if num_runs == 1 else "s"
        times = [o['total_seconds'] for o in runs]
        t = ""
        if num_runs > 0:
            t += f", total time {statistics.mean(times):8.2f}s"
            if num_runs > 1:
                t += f" ± {statistics.stdev(times):7.2f}s"
        print(f"    {commit} {prefix} run{plural} found in S3{t}")

    print()

    times = [o["total_seconds"] for runs in objects.values() for o in runs]
    total_mean = statistics.mean(times)
    total_stdev = statistics.stdev(times)
    print(f"Prior average total time: {total_mean:8.2f}s ± {total_stdev:.2f}s")
    print(f"Current       total time: {obj['total_seconds']:8.2f}s")
    stdevs_bigger = (obj["total_seconds"] - total_mean) / total_stdev
    stdevs_abs = abs(stdevs_bigger)
    stdevs_floor = math.floor(stdevs_abs)
    stdevs_ceil = math.ceil(stdevs_abs)
    if stdevs_abs < stdev_threshold:
        icon, verb, prep, amount = "🟢", "maintains", "within", stdevs_ceil
    else:
        prep, amount = "by at least", stdevs_floor
        if stdevs_bigger < 0:
            icon, verb = "🟣", "reduces"
        else:
            icon, verb = "🔴", "increases"
    plural = "" if amount == 1 else "s"
    print(f"{icon} this commit {verb} total test job time {prep} {amount} standard deviation{plural}")

    all_runs = [obj] + [run for runs in objects.values() for run in runs]
    all_tests = defaultdict(set)
    for run in all_runs:
        for name, suite in run["suites"].items():
            all_tests[name] |= {case["name"] for case in suite["cases"]}

    print()
    print("------ tests added/removed ------")

    for suite, cases in all_tests.items():
        missing_suite = []
        missing_cases = defaultdict(list)
        for commit, runs in objects.items():
            for run in runs:
                suite_dict = run["suites"].get(suite)
                if suite_dict:
                    run_cases = {case["name"] for case in suite_dict["cases"]}
                    for case in cases - run_cases:
                        missing_cases[case].append(commit)
                else:
                    missing_suite.append(commit)
        if missing_suite or missing_cases:
            print()
            print(f"test suite {suite}:")
            if missing_suite:
                print("    missing in these commits:")
                for missing_commit in missing_suite:
                    print(f"        {missing_commit}")
            for case, missing_commits in missing_cases.items():
                print(f"    test case {case} missing in these commits:")
                for missing_commit in missing_commits:
                    print(f"        {missing_commit}")

    print()
    print("--- tests whose times changed ---")

    for suite, cases in all_tests.items():
        info = {}
        curr_suite_dict = obj["suites"].get(suite)
        missing_suite = False
        if not curr_suite_dict:
            for commit, runs in objects.items():
                if runs:
                    missing_suite = True
                for run in runs:
                    if suite in run["suites"]:
                        pass
        for commit, runs in objects.items():
            for run in runs:
                suite_dict = run["suites"].get(suite)
        if info:
            print_suite_regression_info(info)

def positive_integer(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a natural number")
    return parsed

def positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive rational number")
    return parsed

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        "Print statistics from test XML output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--longest-of-class",
        type=positive_integer,
        default=3,
        metavar="N",
        help="how many longest tests to show for each class",
    )
    parser.add_argument(
        "--class-print-threshold",
        type=positive_float,
        default=1.0,
        metavar="N",
        help="Minimal total time to warrant class report",
    )
    parser.add_argument(
        "--longest-of-run",
        type=positive_integer,
        default=10,
        metavar="N",
        help="how many longest tests to show from the entire run",
    )
    parser.add_argument(
        "--upload-to-s3",
        action="store_true",
        help="upload test time to S3 bucket",
    )
    parser.add_argument(
        "--compare-with-s3",
        action="store_true",
        help="download test times for base commits and compare",
    )
    parser.add_argument(
        "--num-prev-commits",
        type=positive_integer,
        default=10,
        metavar="N",
        help="how many previous commits to compare test times with",
    )
    parser.add_argument(
        "--stdev-threshold",
        type=positive_integer,
        default=3,  # to be conservative and reduce false positives
        metavar="s",
        help="minimum standard deviations difference to count as anomaly",
    )
    parser.add_argument(
        "--use-json",
        metavar="FILE.json",
        help="compare S3 with JSON file, instead of the test report folder",
    )
    parser.add_argument(
        "folder",
        help="test report folder",
    )
    args = parser.parse_args()

    reports = parse_reports(args.folder)
    if len(reports) == 0:
        print(f"No test reports found in {args.folder}")
        sys.exit(0)

    send_report_to_scribe(reports)

    longest_tests = []
    total_time = 0
    for name in sorted(reports.keys()):
        test_suite = reports[name]
        if test_suite.total_time >= args.class_print_threshold:
            test_suite.print_report(args.longest_of_class)
        total_time += test_suite.total_time
        longest_tests.extend(test_suite.test_cases)
    longest_tests = sorted(longest_tests, key=lambda x: x.time)[-args.longest_of_run:]

    obj = assemble_s3_object(reports, total_seconds=total_time)

    if args.upload_to_s3:
        send_report_to_s3(obj)

    print(f"Total runtime is {datetime.timedelta(seconds=int(total_time))}")
    print(f"{len(longest_tests)} longest tests of entire run:")
    for test_case in reversed(longest_tests):
        print(f"    {test_case.class_name}.{test_case.name}  time: {test_case.time:.2f} seconds")

    if args.compare_with_s3:
        head_json = obj
        if args.use_json:
            head_json = json.loads(Path(args.use_json).read_text())
        print_regressions(
            head_json,
            n=args.num_prev_commits,
            stdev_threshold=args.stdev_threshold,
        )
