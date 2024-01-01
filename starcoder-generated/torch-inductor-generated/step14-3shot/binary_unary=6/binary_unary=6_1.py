
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = torch.nn.functional.relu(v2)
        return v3
x2 = torch.randn(1, 3)
