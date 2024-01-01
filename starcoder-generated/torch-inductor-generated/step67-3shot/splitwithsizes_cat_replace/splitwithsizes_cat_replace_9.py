
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(1, 32, 1, bias=False), torch.nn.Conv2d(32, 1, 1, bias=False)])
        self.features[0].weight.data = torch.ones((32, 32, 1, 1), dtype=torch.float)
        self.features[1].weight.data = torch.ones((1, 32, 1, 1), dtype=torch.float)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
