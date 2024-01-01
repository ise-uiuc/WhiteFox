
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, 2, 1), torch.nn.Conv2d(32, 3, 1, 1, 0))
        self.split = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 1, 1, 0))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(v1, [1, 1, 1], dim=2)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=2))
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
