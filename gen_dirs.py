import os

targets = (['\\div', '\\pm', '[', ']', '\\log', '\\tan', '\\beta', '\\alpha', 
            '\\int', '\\pi', ',', '\\cos', '\\sum', '\\theta', 'dot', '\\times',
            '\\sin', '\\sqrt', '=', 'rparen', 'lparen', '+', '-'])
os.mkdir('new_train')

for i,t in enumerate(targets):
    os.mkdir('new_train/' + str(i))
    print str(i) + ': ' + t
