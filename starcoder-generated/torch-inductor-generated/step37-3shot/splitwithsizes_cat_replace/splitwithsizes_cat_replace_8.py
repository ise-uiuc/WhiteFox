
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = [32, 32, 32, 32, 64]
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(self.features[i], self.features[i + 1], 3, bias=True) for i in range(len(self.features) - 1)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, list(range(5)))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
