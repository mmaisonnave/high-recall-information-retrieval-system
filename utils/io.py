from IPython.core.display import display, HTML
import datetime
from ipywidgets import Textarea, Button, HBox
from IPython.display import display, clear_output
def info(str_, writer=None):
    print(f'{datetime.datetime.now()} [ \033[1;94mINFO\x1b[0m  ] {str_}')
    if not writer is None:
        writer.write(f'{datetime.datetime.now()} [ INFO  ] {str_}\n')
def ok(str_, writer=None):
    print(f'{datetime.datetime.now()} [  \033[1;92mOK\x1b[0m   ] {str_}')
    if not writer is None:
        writer.write(f'{datetime.datetime.now()} [  OK   ] {str_}\n')
def warning(str_, writer=None):
    print(f'{datetime.datetime.now()} [\x1b[1;31mWARNING\x1b[0m] {str_}')
    if not writer is None:
        writer.write(f'{datetime.datetime.now()} [WARNING] {str_}\n')
def html(str_=''):
    display(HTML(str_))
    
def request_user_input(user_callback):
    t = Textarea()
    b = Button()
    box=HBox([t,b])
    def callback(button=None):
        str_ = t.value
        t.layout.visibility='hidden'
        b.layout.visibility='hidden'
        box.layout.visibility='hidden'
        user_callback(str_)
    b.description='Submit'
    b.on_click(callback)
    display(box)

def ask_confirmation():
    response = input('Are you sure (y/n)? ')
    while response!='n' and response!='y':
        response = input('Invalid response, please try again. Are you sure (y/n)? ')
    assert response=='y' or response=='n'
    return response=='y'
        