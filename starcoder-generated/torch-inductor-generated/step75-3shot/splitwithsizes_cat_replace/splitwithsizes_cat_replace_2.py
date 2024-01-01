
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 8, 3, 1, 1)])
    def forward(self, v0):
        split_tensors = torch.split(v0, [1, 1, 1], dim=1)
        concat1 = torch.cat(split_tensors, dim=1)
        return (concat1,)
# Inputs to the model
x0 = torch.randn(1, 3, 64, 64)
