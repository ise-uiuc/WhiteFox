
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = F.dropout2d(x, p=0.5, training=True)
# Inputs to the model
x = torch.randn(8, 8, 3, 3)
