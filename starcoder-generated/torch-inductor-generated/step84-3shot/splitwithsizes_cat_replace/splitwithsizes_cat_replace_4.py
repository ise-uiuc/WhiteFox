
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        return (split_tensors,)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
