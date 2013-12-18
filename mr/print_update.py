import sys, time
try:
    from IPython.core.display import clear_output
except ImportError:
    pass

def print_update(message):
    try:
        clear_output()
    except Exception:
        pass
    print message
    sys.stdout.flush()
