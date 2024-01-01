
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential()
        self.block1.add_module('conv1', torch.nn.Conv2d(3, 32, 5, 1, 2))
        self.block1.add_module('relu1', torch.nn.ReLU())
        self.block1.add_module('pool1', torch.nn.AvgPool2d(2))
        self.block1.add_module('dropout1', torch.nn.Dropout2d(0.5))
        self.block1.add_module('conv2', torch.nn.Conv2d(32, 128, 3, 1, 1))
        self.block1.add_module('relu2', torch.nn.ReLU())
        self.block1.add_module('pool2', torch.nn.AvgPool2d(2))
        self.block1.add_module('dropout2', torch.nn.Dropout2d(0.5))

        self.block2 = torch.nn.Sequential()
        self.block2.add_module('conv3', torch.nn.Conv2d(128, 256, 3, 1, 1))
        self.block2.add_module('relu3', torch.nn.ReLU())
        self.block2.add_module('pool3', torch.nn.AvgPool2d(2))
        self.block2.add_module('dropout3', torch.nn.Dropout2d(0.5))
        self.block2.add_module('conv4', torch.nn.Conv2d(256, 256, 3, 1, 1))
        self.block2.add_module('relu4', torch.nn.ReLU())
        self.block2.add_module('pool4', torch.nn.AvgPool2d(2))
        self.block2.add_module('dropout4', torch.nn.Dropout2d(0.5))

        self.block3 = torch.nn.Sequential()
        self.block3.add_module('conv5', torch.nn.Conv2d(256, 256, 3, 1, 1))
        self.block3.add_module('relu5', torch.nn.ReLU())
        self.block3.add_module('pool5', torch.nn.AvgPool2d(2))
        self.block3.add_module('dropout5', torch.nn.Dropout2d(0.5))

        self.block4 = torch.nn.Sequential()
        self.block4.add_module('flatten', torch.nn.Flatten())
        self.block4.add_module('fc1', torch.nn.Linear(196608, 2048))
        self.block4.add_module('relu6', torch.nn.ReLU())
        self.block4.add_module('dropout6', torch.nn.Dropout2d(0.5))
        self.block4.add_module('fc2', torch.nn.Linear(4096, 1024))
        self.block4.add_module('relu7', torch.nn.ReLU())
        self.block4.add_module('dropout7', torch.nn.Dropout2d(0.5))
        self.block4.add_module('fc3', torch.nn.Linear(2048, 100))     
    def forward(self, x1):
        v1 = self.block1(x1)    
        v2 = self.block2(v1)
        v3 = self.block3(v2)
        v4 = self.block4(v3)
        return v4
# Inputs to the model
x1 =torch.randn(1, 3, 64, 64)
