
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose1 = torch.nn.Transpose(0, 2)
        self.transpose2 = torch.nn.Transpose(0, 2)
        self.transpose3 = torch.nn.Transpose(0, 2)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.transpose1(x1)
        v2 = self.relu(v1)
        v3 = self.transpose2(v2)
        v4 = self.relu(v3)
        v5 = self.transpose3(v4)
        return self.relu(v5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
