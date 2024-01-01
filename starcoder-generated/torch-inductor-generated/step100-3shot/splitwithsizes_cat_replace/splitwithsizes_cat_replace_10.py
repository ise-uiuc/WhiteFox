
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(MyModule(), MyModule())
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
    def forward(self, x1):
        return (torch.cat([x1]+[x1]*2), torch.split(x1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
