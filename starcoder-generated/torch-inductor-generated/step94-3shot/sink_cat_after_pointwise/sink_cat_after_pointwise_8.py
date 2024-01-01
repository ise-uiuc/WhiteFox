
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if x.shape[0]!= 3:
            x = x.repeat((3, 1))
        return x
# Inputs to the model
x = torch.randn(2, requires_grad=True)
