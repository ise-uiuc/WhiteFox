
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        temp = x.shape[1] + x.shape[2]
        x = x + temp + 1
        return x
# Inputs to the model
x = torch.randn(3, 4, 5)
