import sys
import pytest

def run_all_tests():
    print("=================================================")
    print("Starting JS-MFPCA Test Suite...")
    print("=================================================")
    
    # Arguments passed to the pytest engine:
    # "tests/" : The directory to search for test files
    # "-v"     : Verbose mode (lists every test function by name)
    # "-s"     : Prevents pytest from capturing standard output (allows print statements)
    # "-W ignore" : Ignores deprecation warnings to keep the output clean
    args = ["jsmfpca/tests/", "-v", "-s", "-W", "ignore"]
    
    # Execute pytest programmatically
    exit_code = pytest.main(args)
    
    print("=================================================")
    if exit_code == 0:
        print("All tests passed successfully!")
    else:
        print(f"Test suite finished with exit code {exit_code}.")
    print("=================================================")
    
    # Exit with the appropriate code so CI/CD pipelines know if tests failed
    sys.exit(exit_code)

if __name__ == "__main__":
    run_all_tests()
