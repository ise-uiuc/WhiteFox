
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.relu(x - 0.5527)
        return v1
# Inputs to the model
x = torch.randn(4, 4)
