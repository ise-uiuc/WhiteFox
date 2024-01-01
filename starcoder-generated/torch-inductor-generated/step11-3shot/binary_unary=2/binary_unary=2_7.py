
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 25)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 25 
        v3 = F.sigmoid(v2) 
        return v3
# Inputs to the model
x1 = torch.randn(10,12)
