
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, 1, 1), torch.nn.Conv2d(8, 16, 3, 1, 1), torch.nn.Conv2d(16, 1, 3, 1, 1))
        self.split = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, 1, 1), torch.nn.Conv2d(2, 4, 3, 1, 1), torch.nn.Conv2d(4, 8, 3, 1, 0))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(v1, [1] * 8, dim=2)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
