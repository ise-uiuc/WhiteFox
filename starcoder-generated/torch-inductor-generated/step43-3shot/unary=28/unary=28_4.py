
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, x2, x3, x4)
        return torch.mean(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 224)
x2 = torch.abs(torch.randn(1)[0])
x3 = torch.nn.functional.gelu(torch.abs(torch.randn(1)[0]))
x4 = torch.abs(torch.randn(1)[0])
