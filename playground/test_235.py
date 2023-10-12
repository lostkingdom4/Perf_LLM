import os.path, subprocess
from subprocess import Popen, PIPE, TimeoutExpired

cmd = 'python3 -m py_compile 235.py'
timeout_duration =10
proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,shell=True)
with open('input.0.txt', "r", encoding = 'utf-8') as tf: 
	file_contents = tf.read()
try:
	stdout_data, stderr_data = proc.communicate(input=file_contents.encode('utf-8'), timeout=timeout_duration)
	proc.kill()
	proc.communicate()
	print(stdout_data, stderr_data)
except TimeoutExpired:
	print("timeout")
	proc.kill()  # Terminate the process if timeout occurs
	proc.communicate()  # Ensure to consume the stdout and stderr to avoid deadlocks