"""
Module: SCAL_UI

This module provides an interactive user interface for managing and initiating sessions of SCAL 
(Selective Continuous Active Learning), which is used for labeling data and exploring topics 
in an information retrieval system. The interface is built using `ipywidgets` and integrates 
with Jupyter Notebook.

Classes:
--------
- SCAL_UI:
    An interactive widget-based interface to create, load, or extend SCAL sessions. It provides
    options for session selection, topic description, and keyword management. Users can initiate 
    or resume SCAL processes seamlessly.

Usage:
------
1. Import the SCAL_UI module:
    ```python
    from utils.scal_ui import SCAL_UI
    ```

2. Define a callback function for session initialization:
    ```python
    def start_system(session_name, topic_description):
        # Define logic to handle session startup
        pass
    ```

3. Create an instance of SCAL_UI and pass the callback function:
    ```python
    ui = SCAL_UI(start_system, second_round=True)
    ```

4. Use the UI within a Jupyter Notebook to interact with SCAL sessions.

Features:
---------
- Session Management:
    - Combobox to select existing sessions or create new ones.
    - Automatically filters sessions for specific conditions when `second_round=True`.

- Topic Description:
    - Dynamically updates based on user input.
    - The topic can only be defined with keywords within the vocabulary of `QueryDataItem` (obtained from GloVe vocab).
    - Provides a list of predefined keywords from `QueryDataItem`.

- Keyword Buttons:
    - Allows users to add, display, and remove keywords interactively.

- Button Actions:
    - 'START' or 'LOAD' session based on session availability.
    - Trigger the callback function with session details.

Dependencies:
-------------
- os: For file and directory operations.
- ipywidgets: For building interactive widgets.
- IPython.display: For rendering UI components in Jupyter.
- utils.data_item.QueryDataItem: Provides keyword suggestions and mappings.

"""

import os
import ipywidgets as widgets
from utils.data_item import QueryDataItem
from IPython.display import display, clear_output

# def session_name_ui(callback_fn):
class SCAL_UI(object):
    def __init__(self, callback_fn, second_round=False):
        ensure_option=False
        options = os.listdir('sessions/scal/')
        options = [option for option in options if not option.endswith('_second_round')]
        if second_round:
            def finished_session(session):
                if not os.path.exists(f'sessions/scal/{session}/data'):
                    return False
                datafiles=os.listdir(f'sessions/scal/{session}/data')
                exported= any([datafile.startswith('exported_data') for datafile in datafiles])
                labeled= any([datafile.startswith('labeled_data') for datafile in datafiles])
                return exported and labeled
            options = list(filter(finished_session, options))
            ensure_option=True
        self.session_name_widget = widgets.Combobox(placeholder='Select a saved session or enter new session name',
                                                  options=options,
                                                  description='Session name:',
                                                  ensure_option=ensure_option,
                                                   layout=widgets.Layout(width='425px'),
                                                  style={'description_width': 'initial'},
                                                  disabled=False)


        self.topic_description_widget = widgets.Combobox(placeholder='',
                                                    options=[word for word in QueryDataItem.word2index],
                                                    description='Describe topic here: ',
                                                    ensure_option=False,
                                                    layout=widgets.Layout(width='425px'),
                                                    style={'description_width': 'initial'},
                                                    disabled=False)

        self.topic_description_widget.layout.visibility = 'hidden'

        self.main_button = widgets.Button(description='START', disabled=True, )

        def self_removal(button=None):
            button.layout.visibility='hidden'
            len1=len(self.keyword_buttons)
            self.keyword_buttons = [b for b in self.keyword_buttons if b!=button]
            len2=len(self.keyword_buttons)
            assert len1!=len2
            del(button)               
            clear_output(wait=True)
            self.main_button.disabled = len(self.keyword_buttons)==0
            self.keyword_box=widgets.HBox(self.keyword_buttons)
            display(widgets.VBox([self.session_name_widget,self.topic_description_widget,self.main_button]))
            display(self.keyword_box)
        self.keyword_buttons = []
        self.keyword_box=widgets.HBox(self.keyword_buttons)

        def observe_session_name_widget(widget):
    #         button.disabled = False if len(session_name_text.value)>0 and len(topic_description_text.value)>0  else True
            if os.path.exists(f'sessions/scal/{self.session_name_widget.value}'):
                self.topic_description_widget.layout.visibility = 'hidden'
                self.main_button.disabled=False
                self.main_button.description= 'LOAD'
            else:
                self.main_button.description= 'START'
                self.main_button.disabled=len(self.topic_description_widget.value)==0
                self.topic_description_widget.layout.visibility = 'visible'

        def submit_topic_description_widget(widget):
            if self.topic_description_widget.value in QueryDataItem.word2index:
                button = widgets.Button(description=self.topic_description_widget.value)
                button.on_click(self_removal)
                self.keyword_buttons.append(button)

                self.keyword_box=widgets.HBox(self.keyword_buttons)
                clear_output(wait=True)
                self.topic_description_widget.value=''
                display(widgets.VBox([self.session_name_widget,self.topic_description_widget,self.main_button]))
                display(self.keyword_box)
            self.main_button.disabled = len(self.keyword_buttons)==0

            
        def invoke_callback(button=None):
            callback_fn(self.session_name_widget.value, ' '.join([b.description for b in self.keyword_buttons]))

        self.main_button.on_click(invoke_callback)

        self.session_name_widget.observe(observe_session_name_widget)
        self.topic_description_widget.on_submit(submit_topic_description_widget)

        display(widgets.VBox([self.session_name_widget,self.topic_description_widget,self.main_button]))
        display(self.keyword_box)

