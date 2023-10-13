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



def compile_prog(filepath, lang):
    '''
    filepath: path of the file you would like to compile
    lang: prog. language; 'Py', 'Java', 'CPP', 'C', 'PHP', 'JS', 'CS'
    Dependencies:
    Java: Java Development kit (JDK) (https://www.oracle.com/java/technologies/downloads/)
    JS: Node.js (https://nodejs.org/en/download/)
    CS: Install mono library (brew install mono) (http://www.mono-project.com/Mono:OSX)
    '''
    if lang=='Py':
        cmd = 'taskset -c 4 python3 -m py_compile '+filepath
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
    start_time = time.time()
    #print("start subprocess")
    proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,shell=True)
    #print(proc.pid)
    try:
        #print(file_contents)
        stdout_data, stderr_data = proc.communicate(input=file_contents.encode('utf-8'), timeout=timeout_duration)
        proc.kill()
        proc.communicate()
        end_time = time.time()
        # Decoding the output and error
        op = stdout_data.decode('utf-8')
        err = stderr_data.decode('utf-8')
        #print(op, err)
        #print("end")

    except TimeoutExpired:
        end_time = time.time()
        proc.kill()  # Terminate the process if timeout occurs
        proc.communicate()  # Ensure to consume the stdout and stderr to avoid deadlocks
        err = "Subprocess exceeded time limit and was terminated."
        op = ''
        print(err)

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


