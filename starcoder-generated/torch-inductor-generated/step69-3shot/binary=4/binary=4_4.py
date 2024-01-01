
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 320
        self.linear = torch.nn.Linear(100, out_channels, bias=False)
        self.other = torch.tensor([[1.0], [0.2], [-0.3]], device="cpu")
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
