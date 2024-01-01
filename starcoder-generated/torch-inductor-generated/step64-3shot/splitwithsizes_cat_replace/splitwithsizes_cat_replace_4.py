
class Custom(torch.nn.Module):
    def __init__(self, n, m):
        super(Custom, self).__init__()
        self.n = n
        self.m = m
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.ModuleList([Custom(1, 4)])
        self.extra = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
