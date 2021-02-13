import getpass
import sys
import os
from vbench.api import Benchmark, BenchmarkRunner
from datetime import datetime

USERNAME = getpass.getuser()

if sys.platform == 'darwin':
    HOME = '/Users/%s' % USERNAME
else:
    HOME = '/home/%s' % USERNAME

try:
    import ConfigParser

    config = ConfigParser.ConfigParser()
    config.readfp(open(os.path.expanduser('~/.vbenchcfg')))

    REPO_PATH = config.get('setup', 'repo_path')
    REPO_URL = config.get('setup', 'repo_url')
    DB_PATH = config.get('setup', 'db_path')
    TMP_DIR = config.get('setup', 'tmp_dir')
except:
    REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    REPO_URL = 'git@github.com:danielballan/mr.git'
    DB_PATH = os.path.join(REPO_PATH, 'vb_suite/benchmarks.db')
    TMP_DIR = os.path.join(HOME, 'tmp/vb_mr')

PREPARE = """
python setup.py clean
"""
BUILD = """
python setup.py build_ext --inplace
"""
dependencies = []

START_DATE = datetime(2012, 9, 19) # first full day when setup.py existed

# repo = GitRepo(REPO_PATH)

RST_BASE = 'source'

def generate_rst_files(benchmarks):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    vb_path = os.path.join(RST_BASE, 'vbench')
    fig_base_path = os.path.join(vb_path, 'figures')

    if not os.path.exists(vb_path):
        print('creating %s' % vb_path)
        os.makedirs(vb_path)

    if not os.path.exists(fig_base_path):
        print('creating %s' % fig_base_path)
        os.makedirs(fig_base_path)

    for bmk in benchmarks:
        print('Generating rst file for %s' % bmk.name)
        rst_path = os.path.join(RST_BASE, 'vbench/%s.txt' % bmk.name)

        fig_full_path = os.path.join(fig_base_path, '%s.png' % bmk.name)

        # make the figure
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        bmk.plot(DB_PATH, ax=ax)

        start, end = ax.get_xlim()

        plt.xlim([start - 30, end + 30])
        plt.savefig(fig_full_path, bbox_inches='tight')
        plt.close('all')

        fig_rel_path = 'vbench/figures/%s.png' % bmk.name
        rst_text = bmk.to_rst(image_path=fig_rel_path)
        with open(rst_path, 'w') as f:
            f.write(rst_text)

ref = __import__('benchmarks')
benchmarks = [v for v in ref.__dict__.values() if isinstance(v, Benchmark)]

runner = BenchmarkRunner(benchmarks, REPO_PATH, REPO_URL,
                         BUILD, DB_PATH, TMP_DIR, PREPARE,
                         always_clean=True,
                         run_option='eod', start_date=START_DATE,
                         module_dependencies=dependencies)

if __name__ == '__main__':
    runner.run()
    generate_rst_files(benchmarks)
