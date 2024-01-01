
class Model(torch.nn.Module):
    def __init__(self, in_channels, num_outs=1):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, num_outs)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 1
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(256)

# Inputs to the model
x1 = torch.randn(2, 256)
