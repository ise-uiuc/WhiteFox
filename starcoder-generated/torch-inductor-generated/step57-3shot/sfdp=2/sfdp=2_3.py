
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
 
    def forward(self, x):
        v1 = self.relu(self.conv1(x))
        v2 = self.conv2(v1)
        v3 = self.tanh(v2)
        v4 = self.sigmoid(v1)
        v5 = v1 + v3
        return v4 * v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
