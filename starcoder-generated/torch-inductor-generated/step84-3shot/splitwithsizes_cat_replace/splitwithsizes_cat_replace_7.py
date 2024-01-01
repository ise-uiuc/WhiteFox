
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(6, 32, 3, 1, 1)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return split_tensors[0]
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
