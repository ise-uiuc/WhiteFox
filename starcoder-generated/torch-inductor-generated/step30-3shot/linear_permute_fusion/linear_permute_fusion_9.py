
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.relu(x1)
        v1 = self.relu(x1)
        return v1
# Inputs to the model
x1 = torch.tensor([2.2, -4, -2], dtype=torch.float)
