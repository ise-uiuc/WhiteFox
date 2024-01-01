
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 3, 3, 1, 1))
        self.split = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3, 1, 1), torch.nn.Conv2d(1, 32, 5, 1, 4), torch.nn.Conv2d(1, 1, 3, 1, 2))
    def forward(self, x1):
        v1 = self.features(x1)
        concatenated_tensor = torch.cat(v1, dim=0)
        split_tensors = torch.split(v1, [1, 1, 1], dim=0)
        return (split_tensors, concatenated_tensor)
# Inputs to the model
x1 = torch.randn(15, 3, 64, 64)
x2 = torch.randn(15, 3, 64, 64)
