
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(50, 20)
    def forward(self, x):
        t1 = self.linear(x)
        t2 = torch.tanh(t1)
        return t2.view(20, 5, 5)
# Inputs to the model
x = torch.randn(80, 50)
