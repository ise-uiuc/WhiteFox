
class Model(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels,out_channels)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model(3,5)

# Inputs to the model
x1 = torch.randn(1, 3)
