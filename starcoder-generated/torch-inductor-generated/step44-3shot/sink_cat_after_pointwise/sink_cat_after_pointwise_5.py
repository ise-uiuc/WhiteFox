
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1, y2 = torch.split(x, split_size_or_sections=1, dim=1)
        return y2.view(y2.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
