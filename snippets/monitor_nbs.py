"""
Prints notebook's pid, memory and gpu memory utilization
The code is from https://stackoverflow.com/a/44936664/996379
Also gpu code is from https://gist.github.com/takuseno/2958caf1cb5e74314a9b5971999182b2
Usage:
python monitor_notebooks.py $notebook_token
if token is not supplied, it will prompt you for a token.

src: https://gist.github.com/2torus/dbeb47b81eb18f2baf20bff6adf7b805
"""
import os
import os.path
import posixpath
import subprocess
import sys
import requests as rq

import pandas as pd
import psutil

from subprocess import Popen, PIPE
from xml.etree.ElementTree import fromstring, Element


# this code is from:
# https://gist.github.com/takuseno/2958caf1cb5e74314a9b5971999182b2
def nvidia_smi_xml():
    try:
        p = Popen(['nvidia-smi', '-q', '-x'], stdout=PIPE)
        outs, errors = p.communicate()
        return fromstring(outs)
    except FileNotFoundError:
        return Element('no-gpu')
    

def xml_dict(el, tags):
    return {node.tag: node.text for node in el if node.tag in tags}


def process_gpu_info():
    xml = nvidia_smi_xml()
    data = []
    for gpu_id, gpu in enumerate(xml.iter('gpu')):
        processes = gpu.find('processes').iter('process_info')
        #process_info = [xml_dict(process, ['pid', 'used_memory']) for process in processes]
        process_info = [
            {
            'gpu': str(gpu_id),
            'pid': int(process.findtext('pid')),
            'gpu_memory': process.findtext('used_memory')
        } for process in processes
        ]
        data.extend(process_info)
    return pd.DataFrame.from_records(data, columns=['gpu', 'pid', 'gpu_memory'])



def show_notebooks_table(host, port):
    """Show table with info about running jupyter notebooks.

    Args:
        host: host of the jupyter server.
        port: port of the jupyter server.

    Returns:
        DataFrame with rows corresponding to running notebooks and following columns:
            * index: notebook kernel id.
            * path: path to notebook file.
            * pid: pid of the notebook process.
            * memory: notebook memory consumption in percentage.
    """
    notebooks = get_running_notebooks(host, port)
    prefix = long_substr([notebook['path'] for notebook in notebooks])
    df = pd.DataFrame(notebooks)
    df = df.set_index('kernel_id')
    df.index.name = prefix
    df.path = df.path.apply(lambda x: x[len(prefix):])
    df['pid'] = df.apply(lambda row: get_process_id(row.name), axis=1)
    # same notebook can be run in multiple processes
    df = expand_column(df, 'pid')
    df['pid'] = df['pid'].astype(int)
    df['memory'] = df.pid.apply(memory_usage_psutil)
    df['memory'] = df['memory'].map('{:,.1f}%'.format)
    gpu_df = process_gpu_info()
    df = pd.merge(df, gpu_df, on='pid', how='left')
    return df.sort_values('memory', ascending=False)


def get_running_notebooks(host, port):
    """Get kernel ids and paths of the running notebooks.

    Args:
        host: host at which the notebook server is listening. E.g. 'localhost'.
        port: port at which the notebook server is listening. E.g. 8888.
        username: name of the user who runs the notebooks.

    Returns:
        list of dicts {kernel_id: notebook kernel id, path: path to notebook file}.
    """
    # find which kernel corresponds to which notebook
    # by querying the notebook server api for sessions
    if len(sys.argv) >= 2 and sys.argv[1] is not None:
        token = sys.argv[1]
    else:
        token = input('Specify token: ')
    headers = {'Authorization': f'token {token}'}
    sessions_url = posixpath.join('http://%s:%d' % (host, port), 'api', 'sessions')
    response = rq.get(sessions_url, headers=headers)
    res = response.json()
    notebooks = [{'kernel_id': notebook['kernel']['id'],
                  'path': notebook['notebook']['path']} for notebook in res]
    return notebooks


def get_process_id(name):
    """Return process ids found by (partial) name or regex.

    Source: https://stackoverflow.com/a/44712205/304209.
    >>> get_process_id('kthreadd')
    [2]
    >>> get_process_id('watchdog')
    [10, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61]  # ymmv
    >>> get_process_id('non-existent process')
    []
    """
    child = subprocess.Popen(['pgrep', '-f', name], stdout=subprocess.PIPE, shell=False)
    response = child.communicate()[0]
    return [int(pid) for pid in response.split()]


def memory_usage_psutil(pid=None):
    """Get memory usage percentage by current process or by process specified by id, like in top.

    Source: https://stackoverflow.com/a/30014612/304209.

    Args:
        pid: pid of the process to analyze. If None, analyze the current process.

    Returns:
        memory usage of the process, in percentage like in top, values in [0, 100].
    """
    if pid is None:
        pid = os.getpid()
    process = psutil.Process(pid)
    return process.memory_percent()


def long_substr(strings):
    """Find longest common substring in a list of strings.

    Source: https://stackoverflow.com/a/2894073/304209.

    Args:
        strings: list of strings.

    Returns:
        longest substring which is found in all of the strings.
    """
    substr = ''
    if len(strings) > 1 and len(strings[0]) > 0:
        for i in range(len(strings[0])):
            for j in range(len(strings[0])-i+1):
                if j > len(substr) and all(strings[0][i:i+j] in x for x in strings):
                    substr = strings[0][i:i+j]
    return substr


def expand_column(dataframe, column):
    """Transform iterable column values into multiple rows.

    Source: https://stackoverflow.com/a/27266225/304209.

    Args:
        dataframe: DataFrame to process.
        column: name of the column to expand.

    Returns:
        copy of the DataFrame with the following updates:
            * for rows where column contains only 1 value, keep them as is.
            * for rows where column contains a list of values, transform them
                into multiple rows, each of which contains one value from the list in column.
    """
    tmp_df = dataframe.apply(
        lambda row: pd.Series(row[column]), axis=1).stack().reset_index(level=1, drop=True)
    tmp_df.name = column
    return dataframe.drop(column, axis=1).join(tmp_df)

if __name__ == '__main__':
    print(show_notebooks_table('localhost', 9999))