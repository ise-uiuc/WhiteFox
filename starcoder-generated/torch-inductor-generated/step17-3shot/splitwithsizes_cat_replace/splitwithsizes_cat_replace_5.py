
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 3, 2, 1), torch.nn.Conv2d(32, 32, 3, 1, 1))
        self.concat = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 12, 2, 2))
        self.split = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 0, 1, 0), torch.nn.Conv2d(32, 32, 5, 2, 0), torch.nn.Conv2d(32, 32, 10, 3, 0))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
