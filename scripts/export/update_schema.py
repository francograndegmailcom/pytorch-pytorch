import argparse
import os

from torch._export.serde import schema_check
from yaml import dump, Dumper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="update_schema")
    parser.add_argument(
        "--prefix", type=str, required=True, help="The root of pytorch directory."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the schema instead of writing it to file.",
    )
    parser.add_argument(
        "--force-unsafe",
        action="store_true",
        help="!!! Only use this option when you are a chad. !!! Force to write the schema even if schema validation doesn't pass.",
    )
    args = parser.parse_args()

    assert os.path.exists(
        args.prefix
    ), f"Assuming path {args.prefix} is the root of pytorch directory, but it doesn't exist."

    commit = schema_check.update_schema()

    if os.path.exists(args.prefix + commit.path):
        if commit.result["SCHEMA_VERSION"] < commit.base["SCHEMA_VERSION"]:
            raise RuntimeError(
                f"Schema version downgraded from {commit.base['SCHEMA_VERSION']} to {commit.result['SCHEMA_VERSION']}."
            )

        if commit.result["TREESPEC_VERSION"] < commit.base["TREESPEC_VERSION"]:
            raise RuntimeError(
                f"Treespec version downgraded from {commit.base['TREESPEC_VERSION']} to {commit.result['TREESPEC_VERSION']}."
            )
    else:
        assert args.force_unsafe, "Existing schema yaml file not found, please use --force-unsafe to try again."

    next_version, reason = schema_check.check(commit, args.force_unsafe)

    if next_version is not None and next_version != commit.result["SCHEMA_VERSION"]:
        raise RuntimeError(
            f"Schema version is not updated from {commit.base['SCHEMA_VERSION']} to {next_version}.\n"
            + "Please either:\n"
            + "    1. update schema.py to not break compatibility.\n"
            + "    or 2. bump the schema version to the expected value.\n"
            + "    or 3. use --force-unsafe to override schema.yaml (not recommended).\n "
            + "and try again.\n"
            + f"Reason: {reason}"
        )

    header = (
        "# @" + "generated by " + os.path.basename(__file__).rsplit(".", 1)[0] + ".py"
    )
    header += f"\n# checksum<<{commit.checksum_result}>>"
    payload = dump(commit.result, Dumper=Dumper, sort_keys=False)

    content = header + "\n" + payload

    if args.dry_run:
        print(content)
        print("\nWill write the above schema to" + args.prefix + commit.path)
    else:
        with open(args.prefix + commit.path, "w") as f:
            f.write(content)
