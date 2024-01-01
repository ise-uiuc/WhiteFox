
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 10, (10,15)) # Construct a sample 2D convolution operation
        self.gelu_act = torch.nn.GELU() # Construct a GELU activation
        self.conv2 = torch.nn.Conv1d(20, 10, 10) # Construct a sample 1D convolution operation
        self.softmax = torch.nn.Softmax(1) # Construct a softmax operation
    def forward(self, x):
        v1 = self.conv1(x).permute(0,2,3,1)
        v2 = self.conv1(x).permute(0,2,3,1)
        v3 = self.conv1(x).permute(0,2,3,1)
        res = torch.cat([v1, v2, v3], 2).permute(0,3,1,2)
        res = self.gelu_act(res)
        res = self.conv2(res).permute(0,2,1)
        return self.softmax(res)
# Inputs to the model
x = torch.randn(1, 2, 10, 15)
