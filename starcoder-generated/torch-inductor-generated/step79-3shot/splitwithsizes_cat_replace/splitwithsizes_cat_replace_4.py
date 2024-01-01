
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Linear(10, 10, bias=False), torch.nn.ReLU(), torch.nn.Linear(10, 10, bias=False)])
    def forward(self, split_tensors):
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 10, 3)
