
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 42
        out_channels = 64
        self.linear = torch.nn.Linear(in_channels, out_channels, bias=False)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 + other
        t3 = t2 + 0.25
        t4 = torch.softmax(t3, dim=1) + 0.5
        t5 = torch.sigmoid(t4)
        return t5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 42)
