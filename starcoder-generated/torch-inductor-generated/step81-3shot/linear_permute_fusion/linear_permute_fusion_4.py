
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        return self.batch_norm(x)
# Inputs to the model
x = torch.randn(1, 3, 10, 10, dtype=torch.float16, requires_grad=True)
