
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.ReLU = torch.nn.ReLU()
    def forward(self, x1):
        v1 = torch.softmax(x1, dim=0)
        x2 = self.ReLU(v1)
        y1 = torch.nn.functional.softmax(x2, dim=-1)
        v2 = x1.permute(1, 0)
        x3 = self.ReLU(v2)
        y2 = torch.nn.functional.softmax(x3, dim=-1)
        return y1, y2, v1
# Inputs to the model
x1 = torch.randn(3, 3)
