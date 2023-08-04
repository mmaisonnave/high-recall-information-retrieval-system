from utils.high_recall_information_retrieval import HRSystem
from bs4 import BeautifulSoup




import os
from IPython.display import display, clear_output
import ipywidgets as widgets
from ipywidgets import (
        Button,
        Dropdown,
        HTML,
        HBox,
        VBox,
        IntSlider,
        FloatSlider,
        Textarea,
        Output,
        ToggleButton,
        Checkbox
)

class UI(object):
    data = []
    def __init__(self,debug=False):
        ##############
        # INIT PANEL #
        ##############
        self.from_scratch_radiobuttons = widgets.RadioButtons(options=['saved database (or DP database)', 'from scratch'],
                                                          description='',
                                                          disabled=False
                                                         )
#                                                 Checkbox(value=False,
#                                               description='From scratch',
#                                               disabled=False,
#                                               )

        self.session_name_text = widgets.Combobox(placeholder='Select a saved session or enter new session name',
                                                  options=[elem for elem in os.listdir('sessions') if not elem=='scal'],
                                                  description='Session name:',
                                                  ensure_option=False,
                                                  style={'description_width': 'initial'},
                                                  disabled=False)
#                                     widgets.Text(value='serperi',
#                                               placeholder='Type something',
#                                               description='Session name:',
#                                               disabled=False,
#                                               style={'description_width': 'initial'},
#                                               )



        
        panel_layout = widgets.Layout(flex_flow='column',
                                        align_items='center',
                                        width='50%')
        
        self.init_panel = widgets.GridBox([HBox([self.session_name_text],layout=panel_layout),
                                           HBox([self.from_scratch_radiobuttons],layout=panel_layout),
                                          ], 
                                          layout=widgets.Layout(grid_template_columns="repeat(2, 475px)",
                                                                border='solid 0.1px',
                                                                display='flex',
                                                                align_items='center',
                                                                height='50px'
                                                                ))
        ##############
        # LOOP PANEL #
        ##############
        self.one_iteration_checkbox = Checkbox(value=True,
                                             description='Search in sample only',
                                             style={'description_width': 'initial'},
                                             disabled=False,
                                               layout=widgets.Layout(width='200px'),
                                             )
        
        self.int_sample_size = widgets.IntText(value=10000,
                                               step=1,
                                               description='Sample size:',
                                               disabled=False,
                                               style={'description_width': 'initial'},
                                               layout=widgets.Layout(width='200px'),
                                               )
        
        self.confidence_slider = FloatSlider(value=0.5,
                                             min=0,
                                             max=1.0,
                                             step=0.01,
                                             description='Confidence:',
                                             disabled=False,
                                             continuous_update=False,
                                             orientation='horizontal',
                                             readout=True,
                                             readout_format='.2f',
                                             layout=widgets.Layout(width='237px'),
                                             ) 
        
        self.int_annotation_batch = widgets.IntText(value=10,
                                                    description='Max labeling batch:',
                                                    disabled=False,
                                                    style={'description_width': 'initial'},
                                                    layout=widgets.Layout(width='200px'),
                                                    )
        
        self.loop_panel = widgets.GridBox([HBox([self.int_annotation_batch],layout=panel_layout),
                                           HBox([self.int_sample_size],layout=panel_layout),
                                           HBox([self.confidence_slider],layout=panel_layout),
                                           HBox([self.one_iteration_checkbox],layout=panel_layout),
                                           ], 
                                           layout=widgets.Layout(grid_template_columns="repeat(4, 237px)",
                                                                 border='solid 0.1px',
                                                                 display='flex',
                                                                 align_items='center',
                                                                 height='50px'
                                                                 ))
        
        
#         HBox([self.int_annotation_batch, 
#                                    self.int_sample_size, 
#                                    self.confidence_slider],layout=widgets.Layout(border='solid 0.1px', height='50px', align_items='center'))
        
        self.top_panel = widgets.VBox([self.init_panel, self.loop_panel], )
#         self.top_panel = widgets.VBox([self.init_panel], ) 
        ################
        # BUTTON PANEL #
        ################
        self.debug=debug
        self.send_email_checkbox = Checkbox(value=False,
                                              description='Send exports over email',
                                              disabled=True,
                                                indent=False,
                                           )
        self.include_suggestions_checkbox = Checkbox(value=False,
                                              description='Include suggestions (time consuming)',
                                              disabled=True,
                                              indent=False,
                                              style={'description_width': 'initial'},
                                           )
        self.descriptions = ['INIT', 'LOOP', 'SAVE','REVIEW', 'EXPORT'] #, 'REVIEW']#, 'STATUS']
        self.on_click_functions = [self.on_click_init, 
                                   self.on_click_loop, 
                                   self.on_click_save, 
                                   self.on_click_review,
                                   self.on_click_export,
#                                    self.on_click_status
                                  ]
        self.buttons = [Button() for i in range(len(self.descriptions))]
        self.system = None
        for idx,button in enumerate(self.buttons):
            button.description = self.descriptions[idx]
            button.disabled=False
            button.on_click(self.on_click_functions[idx])
            
        for button in self.buttons:
            if button.description!='INIT':
                button.disabled=True
#         for i in range(len(self.buttons)-1):
#             self.buttons[i+1].disabled=True
        self.bottom_panel = widgets.GridBox(self.buttons+[self.send_email_checkbox,self.include_suggestions_checkbox],
                                 layout=widgets.Layout(grid_template_columns="repeat(4, 150px)",
                                                                border='solid 0.1px',
#                                                                 display='flex',
                                                                justify_content='space-around',
                                                                flex_wrap='wrap',
                                                                align_items='center',
                                                                height='100px',
                                                                ))
                                                                
                                     
                                     
#                                      border='solid 0.1px', height='50px', align_items='center')
#                                 )
        self.status_text = widgets.Label('[STATUS] ')
        self.status_box = HBox([ self.status_text])
        
        self.labeled_count_label = widgets.Label(value='')
        self.unlabeled_count_label = widgets.Label(value='')
        self.stopping_point_label = widgets.HTML(value='')
        
        panel_layout = widgets.Layout( 
                                        align_items='center',
                                        width='50%')
        boxes = [widgets.HBox([widgets.Label(value='Labeled size: '), self.labeled_count_label],layout=panel_layout),
                 widgets.HBox([widgets.Label(value='Unlabeled size: '), self.unlabeled_count_label],layout=panel_layout),
                 widgets.HBox([widgets.Label(value='Est. recall:'), self.stopping_point_label], layout=panel_layout)]
#                  widgets.HBox([widgets.Label(value='<computing stopping point>'), self.stopping_point_label], layout=panel_layout)]
        
        self.count_box = widgets.GridBox(boxes,
                                 layout=widgets.Layout(grid_template_columns="repeat(3, 316px)",
#                                                                 border='solid 0.1px',
                                                                display='flex',
                                                                align_items='center',
                                                                height='50px'))
        
        self.main_frame = VBox([self.top_panel, self.bottom_panel,self.count_box,self.status_box],layout=widgets.Layout(border='solid 0.1px'))
        self.loop_panel.layout.visibility="hidden"
#         for  box in self.loop_panel.children:
#             for elem in box.children:
#                 elem.layout.visible=False

#     def launch_estimator(self):
        
#         def thread_function(widget):
#             estimated = self.system.estimated_recall()*100
#             widget.value=f'Est. recall: {estimated:>5.2f} %'
            
#         # Thread launch linked to widget 
#         x = threading.Thread(target=thread_function, args=(self.stopping_point_label,))
# #         thread_function(self.stopping_point_label)
#         x.start()
    def update_color_estimated_recall(self, color):
        assert color=='red' or color=='green'
        current = BeautifulSoup(self.stopping_point_label.value, 'html.parser').get_text()
        if current=='':
            current='Estimating...'
        if color=='red':
            self.stopping_point_label.value='<mark style="background-color:rgb(255,110,110)">'+current+'</mark>'
        elif color=='green':
            self.stopping_point_label.value='<mark style="background-color:rgb(110,255,110)">'+current+'</mark>'
            
    def update_recall_estimator(self, str):
        self.stopping_point_label.value = f'{str}'
        
    def disable_buttons(self):
        for  box in self.init_panel.children:
            for elem in box.children:
                elem.disabled=True
        for  box in self.loop_panel.children:
            for elem in box.children:
                elem.disabled=True
                
        for button in self.buttons:
            button.disabled=True
        self.include_suggestions_checkbox.disabled=True
        self.include_suggestions_checkbox.disabled=True
    def enable_buttons(self):
        # FINISH FUNCTION
        for button in self.buttons:
            if button.description=='CANCEL':
                button.description='LOOP'
        for  box in self.init_panel.children:
            for elem in box.children:
                elem.disabled=False
        for  box in self.loop_panel.children:
            for elem in box.children:
                elem.disabled=False
                
        if not self.system is None:
            self._update_counters()
        for button in self.buttons:
            button.disabled=False
        self.send_email_checkbox.disabled=False
        self.include_suggestions_checkbox.disabled=False
    def on_click_init(self, button=None):
        self.loop_panel.layout.visibility="visible"
        def print_function(str_):
            self.status_text.value=str_
        
#         self.top_panel = widgets.VBox([self.init_panel, self.loop_panel], )
#         self.main_frame = VBox([self.top_panel, self.bottom_panel,self.count_box,self.status_box],layout=widgets.Layout(border='solid 0.1px'))
        
        clear_output(wait=False)
        self.disable_buttons()
        display(self.main_frame)
#         display(self.right_panel)
#         display(self.bottom_panel)
        self.system = HRSystem(from_scratch=self.from_scratch_radiobuttons.value=='from scratch', 
                               session_name=self.session_name_text.value,
                               finish_function=self.enable_buttons,
                               debug=self.debug,
                               print_function=print_function,
                               ui=self,
                              )  
        self._update_counters()
        self.buttons[0].description='RE-INIT'
#         self.set_estimator_widget(self.stopping_point_label)

    def _update_counters(self):   
        labeled_count = self.system.get_labeled_count()
        unlabeled_count = self.system.get_unlabeled_count()
        relevant_count = self.system.get_relevant_count()
        irrelevant_count = labeled_count-relevant_count
#         self.labeled_count_label.value=value=f'{labeled_count:12,} ({relevant_count:12,} relevant / {irrelevant_count:12,} irrelevant )'
        self.labeled_count_label.value=f'{labeled_count:12,} ({relevant_count:12,} relevant / {irrelevant_count:12,} irrelevant )'
        self.unlabeled_count_label.value = f'{unlabeled_count:12,}'
#         self.launch_estimator()
    def on_click_loop(self,button=None):        
        
        clear_output(wait=False)
        self.disable_buttons()
#         if button.description=='LOOP':
#         button.description='CANCEL'
#         button.disabled=False
        display(self.main_frame)
#         display(self.right_panel)
#         display(self.bottom_panel)
#         self.system.set_confidence_value(self.confidence_slider.value)
        if self.int_sample_size.value<1:
            self.int_sample_size.value=1
        if self.int_annotation_batch.value<1:
            self.int_annotation_batch.value = 1
        self.system.loop(suggestion_sample_size=self.int_sample_size.value, 
                         confidence_value=self.confidence_slider.value,
                         labeling_batch_size=self.int_annotation_batch.value,
                         one_iteration=self.one_iteration_checkbox.value
                        )
#         elif button.description=='CANCEL':
#             clear_output(wait=False)
#             self.system.cancel_loop()
#             button.description='LOOP'
#             self.enable_buttons()
#             display(self.main_frame)
#         self.model_updated=True
        
    def on_click_save(self, button=None):
        clear_output(wait=False)
        self.disable_buttons()
        display(self.main_frame)
#         display(self.left_panel)
#         display(self.right_panel)
#         display(self.bottom_panel)
        self.system.save()
        self.session_name_text.options=[elem for elem in os.listdir('sessions') if not elem=='scal']
        self.enable_buttons()
    def on_click_export(self, button=None):
        clear_output(wait=False)
        self.disable_buttons()
        display(self.main_frame)
#         display(self.left_panel)
#         display(self.right_panel)
#         display(self.bottom_panel)
        self.system.export(suggestion_sample_size=self.int_sample_size.value, 
                           confidence_value=self.confidence_slider.value,
                           send_email=self.send_email_checkbox.value,
                           compute_suggestions=self.include_suggestions_checkbox.value,
                           )
        self.enable_buttons()
    def on_click_review(self, button=None):
        clear_output(wait=False)
        self.disable_buttons()
        display(self.main_frame)
        self.system.review_labeled(how_many=self.int_annotation_batch.value)
        self._update_counters()

    def on_click_status(self, button=None):
        clear_output(wait=False)
        self.disable_buttons()
        display(self.main_frame)
#         display(self.left_panel)
#         display(self.right_panel)
#         display(self.bottom_panel)
        self.system.status()
        self.enable_buttons()

    def run(self):
        display(self.main_frame)
#         display(self.left_panel)
#         display(self.right_panel)
#         display(self.bottom_panel)
