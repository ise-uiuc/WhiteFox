
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.tensor([198.0686])
    def forward(self, x):
        v1 = x + self.op
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 4, 28, 28)
