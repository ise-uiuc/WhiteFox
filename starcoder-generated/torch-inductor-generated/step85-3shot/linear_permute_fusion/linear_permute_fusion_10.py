
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2):
        v3 = x2.permute(1,0)
        return 0.5 * v3
# Inputs to the model
x2 = torch.randn(12, 2)
