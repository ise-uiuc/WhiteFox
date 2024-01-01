
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x):
        t1 = self.relu6(x)
        t2 = torch.tanh(t1)
        t2 = torch.tanh(t2)
        return t2
# Inputs to the model
x = torch.randn(1, 50)
