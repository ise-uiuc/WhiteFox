
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = torch.mm(x1, x1)
        return v1.view(8).add(self.relu(v1))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
