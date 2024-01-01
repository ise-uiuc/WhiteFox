
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256)
 
    def forward(self, x1):
        x1 = F.adaptive_avg_pool2d(x1, output_size=(1, 1))
        x1 = torch.flatten(x1, 1)
        v1 = self.linear(x1)
        v2 = v1 - 0.6
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 256, 256)
