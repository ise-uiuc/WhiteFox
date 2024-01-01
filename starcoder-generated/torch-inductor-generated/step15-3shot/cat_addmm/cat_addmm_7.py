
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
