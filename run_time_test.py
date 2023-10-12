from compiler.terminal_compiler import TerminalCompiler

lang2compiler = {
    "python": TerminalCompiler("Python"),
}

codes = '''x=sorted([eval(input()) for i in range(10)])[:-4:-1]

for i in x:print(i)'''
print(codes)
problem = 'p00001'
result = lang2compiler["python"].execute_code_string(codes,problem)
print(result)