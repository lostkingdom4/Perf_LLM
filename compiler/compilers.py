from subprocess import Popen, PIPE, TimeoutExpired
import os.path, subprocess
import os
import shutil
import re
import json
from tqdm import tqdm
import chardet
import jsonlines 
import tempfile as tfile
import json
import threading
import time
from threading import Thread
import signal
import psutil


def kill_processes_on_core(core_number):
    for proc in psutil.process_iter(attrs=['pid', 'cpu_affinity']):
        affinity = proc.info['cpu_affinity']
        if core_number in affinity:
            try:
                os.kill(proc.info['pid'], signal.SIGKILL)
            except (psutil.NoSuchProcess, PermissionError):
                pass

def compile_prog(filepath, lang,chunk_number):
    '''
    filepath: path of the file you would like to compile
    lang: prog. language; 'Py', 'Java', 'CPP', 'C', 'PHP', 'JS', 'CS'
    Dependencies:
    Java: Java Development kit (JDK) (https://www.oracle.com/java/technologies/downloads/)
    JS: Node.js (https://nodejs.org/en/download/)
    CS: Install mono library (brew install mono) (http://www.mono-project.com/Mono:OSX)
    '''
    if chunk_number != None:
        core_number = chunk_number
    else:
        core_number = 4
    
    if lang=='Py':
        cmd = 'taskset -c {} python3 -m py_compile {}'.format(core_number, filepath)
        #cmd = 'pylint -E ' + filepath
    elif lang=='Java':
        cmd = 'javac '+filepath
    elif lang=='CPP' or lang == 'C':
        cmd = 'g++ -std=c++11 '+ filepath
    # elif lang=='C':
    #     cmd = 'gcc '+filepath
    elif lang=='PHP':
        # cmd = "/home/aneesh/MuST-CoST/vendor/bin/phpstan analyse -l 5 --no-progress " + filepath 
        cmd = 'php -l ' + filepath
        #cmd = 'php -l -d display_errors=on' + filepath
    elif lang=='JS':
        cmd = 'node '+filepath
    elif lang=='CS':
        cmd = 'mcs '+filepath
        #cmd = 'csc '+filepath
    else:
        print('invalid argument')
        return
    proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE,shell=True)
    error = [i.decode('utf-8') for i in proc.stderr.readlines()]
    err = '\n'.join(error)
    output = [i.decode('utf-8') for i in proc.stdout.readlines()]
    op = '\n'.join(output)
    proc.kill()
    return err, op


def execute_prog(filepath, lang, file_contents,chunk_number):
    '''
    filepath: path of the file you would like to compile
    lang: prog. language; 'Py', 'Java', 'CPP', 'C', 'PHP', 'JS', 'CS'
    Dependencies:
    Java: Java Development kit (JDK) (https://www.oracle.com/java/technologies/downloads/)
    JS: Node.js (https://nodejs.org/en/download/)
    CS: Install mono library (brew install mono) (http://www.mono-project.com/Mono:OSX)
    '''
    def handler(signum, frame):
        raise TimeoutError("Command timed out")    
    if chunk_number != None:
        core_number = chunk_number
    else:
        core_number = 4

    if lang=='Py':
        cmd = 'taskset -c {} python3 {}'.format(core_number, filepath)
        #cmd = 'pylint -E ' + filepath
    elif lang=='Java':
        cmd = 'javac '+filepath
    elif lang=='CPP' or lang == 'C':
        cmd = 'g++ -std=c++11 '+ filepath
    # elif lang=='C':
    #     cmd = 'gcc '+filepath
    elif lang=='PHP':
        # cmd = "/home/aneesh/MuST-CoST/vendor/bin/phpstan analyse -l 5 --no-progress " + filepath 
        cmd = 'php -l ' + filepath
        #cmd = 'php -l -d display_errors=on' + filepath
    elif lang=='JS':
        cmd = 'node '+filepath
    elif lang=='CS':
        cmd = 'mcs '+filepath
        #cmd = 'csc '+filepath
    else:
        print('invalid argument')
        return

    timeout_duration = 5  # for example, 10 seconds
    #start_time = time.time()
    #print("start subprocess")
    #signal.signal(signal.SIGALRM, handler)
    #signal.alarm(5)
    #proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,shell=True)
    #print(proc.pid)
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_duration)
        start_time = time.time()
        #print(file_contents)
        #proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, preexec_fn=os.setsid, shell=True)
        proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        stdout_data, stderr_data = proc.communicate(input=file_contents.encode('utf-8'))
        end_time = time.time()
        #proc.kill()
        #proc.communicate()
        # Decoding the output and error
        op = stdout_data.decode('utf-8')
        err = stderr_data.decode('utf-8')
        #print(op, err)
        #print("end")

    except TimeoutError:
        end_time = time.time()
        #kill_processes_on_core(core_number+1)
        #proc.kill()
        subprocess.run(f"pkill -TERM -P {proc.pid}", shell=True)
        #os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        err = "Subprocess exceeded time limit and was terminated."
        op = ''
        #print(err)

    finally:
        signal.alarm(0)  # Reset the alarm
        elapsed_time = end_time - start_time

    return err, op, elapsed_time

def execute_prog_thread(filepath, lang, file_contents):
    '''
    filepath: path of the file you would like to compile
    lang: prog. language; 'Py', 'Java', 'CPP', 'C', 'PHP', 'JS', 'CS'
    Dependencies:
    Java: Java Development kit (JDK) (https://www.oracle.com/java/technologies/downloads/)
    JS: Node.js (https://nodejs.org/en/download/)
    CS: Install mono library (brew install mono) (http://www.mono-project.com/Mono:OSX)
    '''

    def worker(proc, file_contents, results):
        start_time = time.time()
        stdout, stderr = proc.communicate(input=file_contents.encode('utf-8'))
        end_time = time.time()
        elapsed_time = end_time - start_time
        results["stdout"] = stdout
        results["stderr"] = stderr
        results["time"] = elapsed_time

    if lang=='Py':
        cmd = 'taskset -c 4 python3 '+ filepath
        #cmd = 'pylint -E ' + filepath
    elif lang=='Java':
        cmd = 'javac '+filepath
    elif lang=='CPP' or lang == 'C':
        cmd = 'g++ -std=c++11 '+ filepath
    # elif lang=='C':
    #     cmd = 'gcc '+filepath
    elif lang=='PHP':
        # cmd = "/home/aneesh/MuST-CoST/vendor/bin/phpstan analyse -l 5 --no-progress " + filepath 
        cmd = 'php -l ' + filepath
        #cmd = 'php -l -d display_errors=on' + filepath
    elif lang=='JS':
        cmd = 'node '+filepath
    elif lang=='CS':
        cmd = 'mcs '+filepath
        #cmd = 'csc '+filepath
    else:
        print('invalid argument')
        return

    results = {"stdout": None, "stderr": None, "time": 100}
    timeout_duration = 10  # for example, 10 seconds
    start_time = time.time()
    #print("start subprocess")
    #print(cmd)
    proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,shell=True)
    #print(proc.pid)
    thread = Thread(target=worker, args=(proc, file_contents, results))
    thread.start()
    thread.join(timeout=timeout_duration)

    if thread.is_alive():
        #print("Timeout occurred!")
        proc.kill()
        proc.communicate()
        thread.join()

    op = results["stdout"].decode('utf-8')
    err = results["stderr"].decode('utf-8')
    elapsed_time = results["time"]
    #print(err, op, elapsed_time)
    return err, op, elapsed_time


def remove_comments(string, lang):
    if lang == 'Python':
        pattern = "('''[\s\S]*''')|(''[\s\S]*''')"
        string = re.sub(pattern, '', string)
        return re.sub(r'(?m)^ *#.*\n?', '', string)
    else:
        pattern = '\/\*[\s\S]*\*\/'
        pattern2 = '[^:]//.*|/\\*((?!=*/)(?s:.))+\\*/'
        string = re.sub(pattern, '', string)
        string = re.sub(pattern2, '', string)                                              
        return string
    
    
def php_compiler(code_str):
    prefix = '''"<?php '''
    
    code = """echo 'hello world';"""

    suffix = '" | php -l'

    cmd = "echo " +  code_str + code + suffix
    
    print(cmd)
    
    proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE,shell=True)
    
    error = [i.decode('utf-8') for i in proc.stderr.readlines()]
    err = '\n'.join(error)
    output = [i.decode('utf-8') for i in proc.stdout.readlines()]
    op = '\n'.join(output)
    
    return err, op


