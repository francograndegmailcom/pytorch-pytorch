#!/usr/bin/env python3

from typing import Any, List

from export_pytorch_labels import get_pytorch_labels
from gitutils import (
    get_git_remote_name,
    get_git_repo_dir,
    GitRepo,
)
from trymerge import (
    gh_post_pr_comment,
    GitHubPR,
)


def check_labels(pr_labels: List[str]) -> str:
    is_not_user_facing_pr = any(label.strip() == "topic: not user facing" for label in pr_labels)
    if is_not_user_facing_pr:
        return True

    release_labels = [label for label in get_pytorch_labels() if label.lstrip().startswith("release notes:")]

    return any(label.strip() in release_labels for label in pr_labels)


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Check PR labels")
    parser.add_argument("pr_num", type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    try:
        if not check_labels(pr.get_labels()):
            msg = ("Label your PR! If your changes are user facing and "
                   "intended to be a part of release notes, please use a "
                   "`release notes:` label. If not, please add the `topic: "
                   "not user facing` label. For more information, check out "
                   "https://github.com/pytorch/pytorch/wiki/PyTorch-AutoLabel-Bot#why-categorize-for-release-notes-and-how-does-it-work.")
            gh_post_pr_comment(pr.org, pr.project, pr.pr_num, msg)
    except Exception as e:
        pass


if __name__ == "__main__":
    main()
