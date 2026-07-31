import sys
import pytest


def run_all_tests():
    print("=================================================")
    print("Starting JS-MFPCA Test Suite...")
    print("=================================================")

    args = ["jsmfpca/tests/", "-v", "-s", "-W", "ignore"]
    exit_code = pytest.main(args)

    print("=================================================")
    if exit_code == 0:
        print("All tests passed successfully!")
    else:
        print(f"Test suite finished with exit code {exit_code}.")
    print("=================================================")

    sys.exit(exit_code)


if __name__ == "__main__":
    run_all_tests()
