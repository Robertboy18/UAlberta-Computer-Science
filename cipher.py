# Initialize program in a main function
def main():
    # Set programend value to cover error in file input
    PROGRAM_END = 1
    # Open a file with the data needed
    try:
        # file_name = input('Enter the input filename: ')
        input_file = open("secretMessage1.txt", 'r')
    # If a file is unable to be found, end program
    except IOError:
        print('File not available, end of program')
        PROGRAM_END = 0
    # If the file is found, \n and '' are removed to get individual letters
    while PROGRAM_END == 1:
        code_word = ''
        input_file_1 = input_file.read()
        split_line = input_file_1.split('\n')
        print(split_line)
        key = int((split_line[0])) % 26
        code_word = ''
        for i in split_line[1]:
            i = ord(i) - key
            if i > ord('Z'):
                i -= 26
            elif i < ord('A'):
                i += 26
            elif i < ord('a'):
                i += 26
            elif i > ord('z'):
                i -= 26
            code_word += str(chr(i))
        print(code_word)
    input_file.close
    PROGRAM_END = 0


main()
