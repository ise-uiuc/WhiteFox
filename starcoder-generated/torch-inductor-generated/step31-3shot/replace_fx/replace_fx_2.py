
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x:torch.Tensor):
        return x.view(1, 2, 3),
# Inputs to the model
x = torch.randn(2, 3)
