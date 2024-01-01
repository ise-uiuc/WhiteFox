
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 16, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = v1 + x2
        return v2

# Initializing the model and input tensors
m = Model()
x1 = torch.randn(1, 8)
x2 = torch.ones(1, 16)
