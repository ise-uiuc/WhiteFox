
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.ReLU(), torch.nn.Linear(3, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
