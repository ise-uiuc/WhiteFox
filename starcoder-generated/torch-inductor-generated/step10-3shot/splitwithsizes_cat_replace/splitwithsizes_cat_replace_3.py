
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.ReLU(inplace=False), torch.nn.Hardtanh(-0.10000000000000001, 0.10000000000000001))
        self.split = torch.nn.Sequential(torch.nn.Sigmoid(), torch.nn.ReLU(inplace=True))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(v1, [1, 1, 1, 1], dim=2)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1, 1], dim=2))
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
