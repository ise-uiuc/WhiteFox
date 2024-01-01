
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.conv3=nn.Conv2d(64,128,3,padding=1)
        self.conv4=nn.Conv2d(128,128,3,padding=1)
        self.conv5=nn.Conv2d(128,128,3,padding=1)
        self.conv6=nn.Conv2d(128,256,3,padding=1)

    def forward(self, x1):
        conv1_out =F.relu(self.conv1(x1))
        conv2_out =F.relu(self.conv2(conv1_out))
        conv3_out =F.relu(self.conv3(conv2_out))
        conv4_out =F.relu(self.conv4(conv3_out))
        conv5_out =F.relu(self.conv5(conv4_out))
        #conv6_out =F.relu(self.conv6(conv5_out))
        return conv5_out, conv5_out
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
