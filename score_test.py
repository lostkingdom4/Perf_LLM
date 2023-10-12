from compiler.terminal_compiler_test import TerminalCompiler

lang2compiler = {
    "python": TerminalCompiler("Python"),
}

codes = "x=sorted([eval(input()) for i in range(10)])[:-4:-1]\n\nfor i in x:print(i)"

a,b,did_compile = lang2compiler["python"].compile_code_string(codes)
print(a,b,did_compile)
compiler_path = './playground'
with open(compiler_path + "/output.0.txt", "r", encoding = 'utf-8') as tf: 
    file_contents = tf.read()
print(type(file_contents))
if b == file_contents:
    print("True")