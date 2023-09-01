import sys
import subprocess
import re
import argparse


def parse_diff_output(changed_files):
    # Regex to capture filename and the line numbers of the changes
    file_pattern = re.compile(r"^\+\+\+ b/(.*?)$", re.MULTILINE)
    line_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)

    files = {}
    for match in file_pattern.finditer(changed_files):
        file_name = match.group(1)

        # Filtering for C/C++ files and excluding certain directories
        if file_name.endswith((".c", ".cpp", ".h", ".hpp")) and all(
            excluded not in file_name
            for excluded in [
                "doc/",
                "test_utils/",
                "test/",
                "tests/",
                "unit_test",
                "perf_test",
                "example/",
                "examples/",
            ]
        ):
            # Find the lines that changed for this file
            lines_start_at = match.end()
            next_file_match = file_pattern.search(changed_files, pos=match.span(0)[1])

            # Slice out the part of the diff that pertains to this file
            file_diff = changed_files[
                lines_start_at : next_file_match.span(0)[0] if next_file_match else None
            ]

            # Extract line numbers of the changes
            changed_lines = []
            for line_match in line_pattern.finditer(file_diff):
                start_line = int(line_match.group(1))
                num_lines = int(line_match.group(2) or 1)

                # The start and end positions for this chunk of diff
                chunk_start = line_match.end()
                next_chunk = line_pattern.search(file_diff, pos=line_match.span(0)[1])
                chunk_diff = file_diff[
                    chunk_start : next_chunk.span(0)[0] if next_chunk else None
                ]

                lines = chunk_diff.splitlines()
                line_counter = 0
                for line in lines:
                    if line.startswith("+"):
                        line_counter += 1
                        if (
                            "MPI_COMM_WORLD" in line
                            and not "CHECK: ALLOW MPI_COMM_WORLD" in line
                        ):
                            # Only include lines where "MPI_COMM_WORLD" is added
                            # and "CHECK: ALLOW MPI_COMM_WORLD" is not present
                            changed_lines.append(start_line + line_counter)

            if changed_lines:
                files[file_name] = changed_lines

    return files


def get_changed_files_uncommitted():
    """Get a dictionary of files and their changed lines where MPI_COMM_WORLD was added from uncommitted changes."""
    cmd = ["git", "diff", "-U0", "--ignore-all-space", "HEAD"]
    result = subprocess.check_output(cmd).decode("utf-8")

    return parse_diff_output(result)


def get_changed_files(start_commit, end_commit):
    """Get a dictionary of files and their changed lines between two commits where MPI_COMM_WORLD was added."""
    cmd = ["git", "diff", "-U0", "--ignore-all-space", start_commit, end_commit]
    result = subprocess.check_output(cmd).decode("utf-8")

    return parse_diff_output(result)


def print_occurences(changed_files, title):
    print(title)
    for file_name, lines in changed_files.items():
        print("-----")
        print(f"File: {file_name}")
        print("Changed Lines:", ", ".join(map(str, lines)))
        print("-----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base", default="origin/develop", help="BASE commit (default: %(default)s)"
    )
    parser.add_argument(
        "--head", default="HEAD", help="HEAD commit (default: %(default)s)"
    )

    start_commit = parser.parse_args().base
    print(f"Start commit: {start_commit}")

    end_commit = parser.parse_args().head
    print(f"End commit: {end_commit}")

    commited_occurences = get_changed_files(start_commit, end_commit)
    uncommited_occurences = get_changed_files_uncommitted()

    mpi_comm_world_detected = commited_occurences or uncommited_occurences

    if mpi_comm_world_detected:
        if commited_occurences:
            print_occurences(
                commited_occurences, "Detected MPI_COMM_WORLD in the following files:"
            )
        if uncommited_occurences:
            print_occurences(
                uncommited_occurences,
                "Detected MPI_COMM_WORLD in the following files (uncommited changes):",
            )

        sys.exit(1)  # Exit with an error code to fail the GitHub Action
    else:
        print("No addition of MPI_COMM_WORLD detected.")
        sys.exit(0)
