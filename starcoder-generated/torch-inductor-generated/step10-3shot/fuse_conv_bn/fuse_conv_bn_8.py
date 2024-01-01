
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        output = F.conv2d(F.batch_norm(x, ), )
# Inputs to the model
x = torch.randn(1024, 2, 7, 7)
