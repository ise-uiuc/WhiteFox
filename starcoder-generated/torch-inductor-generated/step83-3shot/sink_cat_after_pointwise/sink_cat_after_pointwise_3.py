
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.relu(x.view(-1, 2, x.size(-1)))
        return y
# Inputs to the model
x = torch.randn(2, 4, 2)
