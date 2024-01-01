 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = torch.nn.Linear(1000, 10)

    def forward(self, t1):
        v1 = self.linear(t1)
        v2 = torch.nn.ReLU()(v1)
        return v2

# Inputs to the model
x1 = torch.randn(1, 1000)
