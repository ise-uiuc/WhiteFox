
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x / 2.0
        x = x.unsqueeze(0).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
