
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1, bias=True), torch.nn.BatchNorm2d(32), torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Conv2d(32, 32, 3, 1, 1, bias=True)])
    def forward(self, v1):
        split_tensors = [torch.split(v1, [s], 1) for s in [3, 1, 1, 1, 1]]
        concatenated_tensor = torch.cat(split_tensors[0], 1)
        return (concatenated_tensor, split_tensors)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
