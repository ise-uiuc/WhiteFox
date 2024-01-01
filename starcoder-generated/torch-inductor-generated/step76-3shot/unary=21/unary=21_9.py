
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # [START] TODO: Define the layers of your model
        self.conv1 = torch.nn.Conv2d(1, 128, 7, stride=2, padding=3) #stride 2, padding 3, output: 128x14x14
        self.dpt1 = torch.nn.Dropout(p = 0.2)
        self.bn1 = torch.nn.BatchNorm2d(128) #output: 128x14x14
        self.mp1 = torch.nn.MaxPool2d(kernel_size = 2,
                                    stride = 2) #output: 128x7x7
        self.conv2 = torch.nn.Conv2d(128, 256, 4, stride=2, #stride 2, padding 1
                                    padding=1) #output: 256x5x5
        self.bn2 = torch.nn.BatchNorm2d(256) #output: 256x5x5
        self.mp2 = torch.nn.MaxPool2d(kernel_size = 2,
                                    stride = 2) #output: 256x2x2
        self.conv3 = torch.nn.Conv2d(256, 512, 5, stride=1, padding=2) #stride 1, padding 2, output: 512x3
        self.bn3 = torch.nn.BatchNorm2d(512) #output: 512x3
        # [END] TODO: Define the layers of your model
    def forward(self, x):
        # [START] TODO: Define the computation performed at every call.
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.dpt1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.tanh(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.tanh(x)
        
        # [END] TODO: Define the computation performed at every call.
        return x
# Inputs to the model
x = torch.randn(128, 1, 28, 28)
