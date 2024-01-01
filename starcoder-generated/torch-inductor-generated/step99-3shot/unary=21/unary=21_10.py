
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 7, padding=(3, 2), dilation=(2, 1))
        self.conv2 = torch.nn.Conv2d(32, 32, 1, padding=(1, 1), dilation=(1, 1))
        self.conv3 = torch.nn.Conv2d(256, 16, 1)
        self.fc1 = torch.nn.Linear(288, 32)
        self.fc2 = torch.nn.Linear(32, 2)
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x) # Should have same padding to match the original convolution operation on Torch
        p1 = torch.cat([c1, c2], dim=1)
        c3 = self.conv3(p1) # Should have (input_padding[0] * 2, input_padding[1] * 2) to match the original convolution operation on Torch
        flattened = torch.flatten(c3, 1)
        fc1 = torch.tanh(self.fc1(flattened))
        fc2 = torch.tanh(self.fc2(fc1))
        return fc2, p1
    # Inputs to the model
    x = torch.randn(1, 3, 224, 224)
    