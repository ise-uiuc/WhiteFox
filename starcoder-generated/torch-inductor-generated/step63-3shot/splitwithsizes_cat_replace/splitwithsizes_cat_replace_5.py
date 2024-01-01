
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([Model()])
    def forward(self, v1):
        split_tensors = torch.split(self.features[0](v1), [1, 1, 1], dim=1)
        return (torch.cat(split_tensors, dim=1), torch.split(self.features[0](v1), [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
