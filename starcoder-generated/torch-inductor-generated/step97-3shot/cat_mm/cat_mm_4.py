
class Model(torch.nn.Module):
    def __init__(self, batch, channel, height, width):
        super().__init__()
        # 1. Linear operation using a randomly generated tensor weight and a constant bias
        # 2. BatchNorm2d
        # 3. ReLU6 activation
        # 4. Reshaping
        # 5. Matrix multiplication
        # 6. Concation
# Inputs to the model
        self.conv1 = torch.nn.Conv2d(
            channel, 
            channel, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            dilation=1, 
            groups=1, 
            bias=False
        )

        self.batchnorm1 = torch.nn.BatchNorm2d(channel)
        self.relu6 = torch.nn.ReLU6(inplace=True)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(batch * 64 * height * width, 1)

        self.batchnorm2 = torch.nn.BatchNorm2d(channel)
        self.relu6 = torch.nn.ReLU6(inplace=True)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(batch * 128 * height * width, 1)
        
    def forward(self, x):
        x = self.conv1(x) # 1
        x = self.batchnorm1(x) # 2
        x = self.relu6(x) # 3
        x = self.flatten(x) # 4
        x = self.linear1(x) # 5

        x = self.batchnorm2(x) # 6
        x = self.relu6(x) # 7
        x = self.flatten(x) # 8
        x = self.linear1(x) # 9
        return x
# Inputs to the model            
        x = torch.randn(1, 64, 56, 56)
