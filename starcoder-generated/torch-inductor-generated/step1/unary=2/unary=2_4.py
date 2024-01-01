
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, output_padding=0)
 
    def forward(self, x):
        v1 = torch.nn.functional.pad(x, (1, 1, 1, 1))
        v2 = self.conv(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7978845608028654
        v5 = v2 * 0.044715
        v6 = v2 * v5
        v7 = v4 + v6
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
