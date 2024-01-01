
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.reshape(x.shape)[0]
        y = x.flatten(start_dim=0)
        z = y.permute(1, 2, 0)
        x = y - z
        return x
# Inputs to the model
x = torch.randn(3, 2, 2)
