
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1, x2):
        t1 = self.linear(x1)
        return t1 + x2
 
# Inputs to the model
x1 = torch.randn(128, 16)
x2 = torch.randn(128, 32)
