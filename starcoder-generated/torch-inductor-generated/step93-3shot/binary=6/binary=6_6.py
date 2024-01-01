
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 9, bias=True)
        
    def forward(self, x1):
        x2 = x1.clone()
        x3 = torch.exp(x2)
        v1 = self.fc(x3)
        v2 = v1 - 35
        return v2

# Inputs to the model
x1 = torch.randn(2, 10)
