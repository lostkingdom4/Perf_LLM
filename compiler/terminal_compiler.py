from .compilers import compile_prog, execute_prog, execute_prog_thread
import tempfile as tfile
import re
import os

class TerminalCompiler:
    
    def __init__(self, language):
        
        self.lang = language
        
        self.lang2ext = {
                'Python' : '.py',
                'C' : '.c',
                'Java': '.java',
                'PHP': '.php',
                'C++': '.cpp',
                'C#': '.cs'
                }
    
        self.lang2compiler = {
                'Python' : 'Py',
                'C' : 'C',
                'Java': 'Java',
                'PHP': 'PHP',
                'C++': 'CPP',
                'C#': 'CS'
                }
        
    def remove_special_tokens(self, code_string):
        lines = code_string.split("NEW_LINE")
        lines = [item.strip() for item in lines]
    
        curr_indent = 0
        new_lines = []
        for line in lines:
            indent_count = line.count('INDENT')
            dedent_count = line.count('DEDENT')
            curr_indent += indent_count - dedent_count
            wo_indent = re.sub('INDENT\s?', '', line)
            wo_dedent = re.sub('DEDENT\s?', '', wo_indent)
            new_lines.append('\t'*curr_indent + wo_dedent)
        return ("\n").join(new_lines)

    def remove_newline(self, code_string):
        return re.sub('NEW_LINE\s?', '\n', code_string)

    def process_php_string(self, code_string):

        
        if code_string.startswith('< ? php'):
            code_string = code_string.replace('< ? php', "<?php")
        elif not code_string.startswith('<?php'):
            code_string = "<?php " + code_string
            
        code_string = code_string.strip()

        if code_string.endswith('? >'):
            code_string = code_string[:-3] + '?>'
        code_string = code_string.replace('$ ', '$')
        return code_string 
        
        
    def compile_code_string(self, code_string, problem, print_error = False, compiler_path = './testfield/'):
        
        if self.lang == 'Python':
            #code_string = self.remove_special_tokens(code_string)
            pass
        else:
            code_string = self.remove_newline(code_string)
            
        if self.lang == 'PHP':
            code_string = self.process_php_string(code_string)
        elif self.lang == "Java":
            code_string = code_string.replace("public class", "class")
            
        # fd, path = tfile.mkstemp(suffix=self.lang2ext[self.lang]) #can use anything 
        # try:
        #     with os.fdopen(fd, 'w') as tmpo:
        #         # do stuff with temp file
        #         tmpo.write(code_string)
        #         tmpo.flush()
        #         print(path)
        #         error, output = compile_prog(path, self.lang2compiler[self.lang])
        # finally:
        #     os.remove(path)
        
        '''
        with tfile.NamedTemporaryFile(mode="w+",suffix=self.lang2ext[self.lang], delete=True, encoding = 'utf-8') as tf:
                tf.write(code_string)
                tf.flush()
                file_path=tf.name
                error, output = compile_prog(file_path, self.lang2compiler[self.lang])
        '''
        file_path = compiler_path + problem + self.lang2ext[self.lang]

        with open(file_path, "w+", encoding = 'utf-8') as tf: 
            tf.write(code_string)

        error, output = compile_prog(file_path, self.lang2compiler[self.lang])
        
        if print_error:
            print("Error: ", error)

        if self.lang == "PHP":
            if "Errors parsing" in output:
                    return error, output, False

            elif "No syntax errors" in output:
                return error, output, True
            # if "[ERROR]" in output:
            #     return error, output, False
            # elif "[OK] No errors" in output:
            #     return error, output, True

        if error:
            return error, output, False
        else:
            return error, output, True


    def execute_code_string(self, code_string, problem, print_error = False, execute_path = './testfield/'):

        def extract_number(filename):
            return int(filename.split('.')[1])
        
        if self.lang == 'Python':
            #code_string = self.remove_special_tokens(code_string)
            pass
        else:
            code_string = self.remove_newline(code_string)
            
        if self.lang == 'PHP':
            code_string = self.process_php_string(code_string)
        elif self.lang == "Java":
            code_string = code_string.replace("public class", "class")
            
        # fd, path = tfile.mkstemp(suffix=self.lang2ext[self.lang]) #can use anything 
        # try:
        #     with os.fdopen(fd, 'w') as tmpo:
        #         # do stuff with temp file
        #         tmpo.write(code_string)
        #         tmpo.flush()
        #         print(path)
        #         error, output = compile_prog(path, self.lang2compiler[self.lang])
        # finally:
        #     os.remove(path)
        
        '''
        with tfile.NamedTemporaryFile(mode="w+",suffix=self.lang2ext[self.lang], delete=True, encoding = 'utf-8') as tf:
                tf.write(code_string)
                tf.flush()
                file_path=tf.name
                error, output = compile_prog(file_path, self.lang2compiler[self.lang])
        '''

        file_path = execute_path + problem + self.lang2ext[self.lang]
        
        with open(file_path, "w+", encoding = 'utf-8') as tf: 
            tf.write(code_string)
            
        directory_path = './data/public_test_cases/' + problem + '/'
        #print(directory_path)
        all_items = os.listdir(directory_path)
        input_files = sorted([f for f in all_items if 'input' in f], key=extract_number)
        output_files = sorted([f for f in all_items if 'output' in f], key=extract_number)
        result = True
        error = False
        if len(input_files) != len(output_files):
            raise ValueError('Number of input files is not equal to the number of output files')

        elapsed_time_accumulate = 0
        for index, file in enumerate(input_files):
            #print(file)
            with open(directory_path + file, "r", encoding = 'utf-8') as tf: 
                file_contents = tf.read()

            err, output, elapsed_time = execute_prog(file_path, self.lang2compiler[self.lang],file_contents)
            elapsed_time_accumulate += elapsed_time
            with open(directory_path + output_files[index], "r", encoding = 'utf-8') as tf: 
                file_contents = tf.read()
            if output != file_contents:
                result = False
            if err:
                error = True


        if print_error:
            print("Error: ", error)

        if self.lang == "PHP":
            if "Errors parsing" in output:
                    return error, output, False

            elif "No syntax errors" in output:
                return error, output, True
            # if "[ERROR]" in output:
            #     return error, output, False
            # elif "[OK] No errors" in output:
            #     return error, output, True

        if error:
            return err, output, False, elapsed_time_accumulate
        else:
            return err, output, True, elapsed_time_accumulate





        
    def compile_code_file(self, file_path, print_error = False):
        
#         if self.lang == 'Python':
#             code_string = self.remove_special_tokens(code_string)
#         else:
#             code_string = self.remove_newline(code_string)
            
#         if self.lang == 'PHP':
#             code_string = self.process_php_string(code_string)
        
#         with tfile.NamedTemporaryFile(mode="w+",suffix=self.lang2ext[self.lang], delete=True) as tf:

#                 tf.write(code_string)
#                 tf.flush()
#                 file_path=tf.name

        error, output = compile_prog(file_path, self.lang2compiler[self.lang])
        
        if print_error:
            print("Error: ", error)

        if error:
            return error, output, False
        else:
            return error, output, True

    