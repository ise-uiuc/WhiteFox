
class Model(torch.nn.Module):
    def __init__(self, output_w, output_h, channels, other):
        super().__init__()
        self.linear1 = torch.nn.Linear(channels, 3)
        self.linear2 = torch.nn.Linear(output_w * output_h * 3, 1)
        self.other = other
    
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 + self.other
        v3 = self.linear2(v2)
        return torch.sigmoid(v3)

# Initializing the model
w = 64
h = 64
channels = 3
other = torch.randn(1, 3, 64, 64)
m = Model(w, h, channels, other)

# Inputs to the model
x = torch.randn(1, channels, w, h)
