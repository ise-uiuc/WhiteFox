
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        other = kwargs["conv-relu_other"] if "conv-relu_other" in kwargs else -0.3
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv_relu_other = torch.nn.Parameter(torch.tensor(other))
 
    def forward(self, x):
        v0 = self.conv_relu_other
        v1 = self.conv(x)
        v2 = v1 - v0
        v3 = self.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
i1 = torch.empty(1, dtype=torch.float64)
