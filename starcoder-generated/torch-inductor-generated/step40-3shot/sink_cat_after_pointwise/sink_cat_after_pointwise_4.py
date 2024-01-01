
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.randn(3, x.shape[0], x.shape[1], x.shape[2], x.shape[2], dtype=torch.double)
        x.to(dtype=torch.float)
        return x[2, :, :, :, 0, 0]
# Inputs to the model
x = torch.randn(5, 5, 5)
