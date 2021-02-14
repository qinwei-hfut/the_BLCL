# 在这个类中，我首先加载原始的PreResNet(应当也兼容原始ResNet)，然后剔除当前的头，添加我们的1，2，3，4,5个头
# 然后在model.py中添加这个不同头的模型；
import torch
import torch.nn as nn
import pdb

class Multi_FC_Model(nn.Module):
    def __init__(self,model,num_classes,block_expansion=1,num_fc=2):
        '''
        model就是我们传入的cnn+fc模型，可以剔除其fc，然后根据我们的需要添加单个或者多个fc
        '''
        super(Multi_FC_Model,self).__init__()
        self.num_fc = num_fc
        self.CNN = nn.Sequential(*list(model.children())[:-1])
        self.fc_list = nn.ModuleList()
        # pdb.set_trace()
        for i in range(self.num_fc):
            self.fc_list.append(nn.Linear(512*block_expansion,num_classes))
        # self.train_FLAG = True
    
    def forward(self,x):
        x = self.CNN(x)

        x = torch.nn.functional.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        outputs = []
        # pdb.set_trace()
        for fc in self.fc_list:
            outputs.append(fc(x))
        return outputs


    # def eval(self):
    #     super().eval()
    #     self.train_FLAG = False
    
    # def train(self):
    #     super().train()
    #     self.train_FLAG = True
    



