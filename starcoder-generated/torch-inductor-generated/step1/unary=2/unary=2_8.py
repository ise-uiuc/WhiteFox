
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.5
        v3 = self.relu(v2 + self.conv(v1))
        v4 = v1 * 0.7071067811865476
        v5 = v1 * 0.044715
        v6 = v3 * 0.7978845608028654
        v7 = v3 + self.conv(v1 * v5 * v4)
        v8 = torch.tanh(v7)
        v9 = v2 * v8
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
