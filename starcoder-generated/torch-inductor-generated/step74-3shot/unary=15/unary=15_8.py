
class model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 63, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(63, 63, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(63, 1, 1, stride=1, padding=1)
        self.transpose = torch.nn.ConvTranspose2d(63, 63, (5,8), stride=2, dilation=2, output_padding=1)
        self.conv7 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(3,stride=1, padding=1)
        self.pad = torch.nn.ConstantPad2d((0, 0, 1, 1), 0.)
        
        self.adaptiveavgpool = torch.nn.AdaptiveAvgPool2d((244, 122))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)  
        v5 = self.relu(v4) 
        v6 = self.conv5(v5)
        v7 = self.conv6(v6)
        v8 = self.transpose(v2)
        v9 = self.pad(v1)
        v10 = self.conv7(v8)
        v11 = torch.relu(v10)
        v12 = self.maxpool(v9)
        v13 = self.adaptiveavgpool(v12)
        return v13 

x = torch.randn(1, 3, 64, 64)
