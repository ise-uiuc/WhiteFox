
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = torch.nn.Linear(67, 33)
 
    def forward(self, x1):
        v1 = self.Linear(x1)
        v2 = v1 - other
        return 
x1 = torch.randn(1, 67)
