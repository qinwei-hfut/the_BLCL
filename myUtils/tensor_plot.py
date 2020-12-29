import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties  
from pylab import *
import random

class TensorPlot():
    def __init__(self,saved_path):
        self.saved_path = saved_path
        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path)
        # self.epoch_list = []
        self.data = {}
        self.color_list = ['b','g','r','c','m','y','k','w',]

    def add_scaler(self,fig_key,value,epoch):
        if epoch not in self.data.keys():
            self.data[epoch] = {}
        self.data[epoch][fig_key] = value
        # self.epoch_list.append(epoch)

    def add_scalers(self,fig_key, value_dict, epoch):
        if epoch not in self.data.keys():
            self.data[epoch] = {}
        self.data[epoch][fig_key] = value_dict
        # self.epoch = epoch

    def flush(self):
        for fig_key, fig_value in self.data[0].items():
            plt.figure(figsize=(6,6))
            if isinstance(fig_value,dict):
                lines_dict = {}
                for k,v in fig_value.items():
                    lines_dict[k] = []
                x_epoch = [] 
            
                for key_epoch, value in self.data.items():
                    x_epoch.append(key_epoch)
                    for line_name, line_value in value[fig_key].items():
                        lines_dict[line_name].append(line_value)

                for idx,(line_name, line_values) in enumerate(lines_dict.items()):
                    plt.plot(x_epoch, line_values, color=self.color_list[idx],linewidth=2.0,label=line_name)

                # 添加figure的setting
                plt.xlabel('epoch')
                plt.legend()
                # plt.ylabel('loss',fontdict=font)
            else:
                line_value = []
                line_name = fig_key
                x_epoch = []

                for key_epoch, value in self.data.items():
                    x_epoch.append(key_epoch)
                    line_value.append(value[fig_key])
                
                plt.plot(x_epoch, line_value, color=self.color_list[random.randint(0,7)], linewidth=2.0, label=line_name)

                plt.xlabel('epoch')
                plt.legend()

            print(os.path.join(self.saved_path,fig_key+'.pdf'))
            plt.savefig(os.path.join(self.saved_path,fig_key+'.pdf'),format='pdf')
            plt.close()
                


        