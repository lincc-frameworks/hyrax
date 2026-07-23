from pathlib import Path

def track_test():
    """This test function should be able to detect by asv and run along for saving results"""
    if Path("./gondor_results").is_dir():
        return(10000000000000)
    else:
        return(99999999999990)

