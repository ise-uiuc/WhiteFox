
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 1, 1, 0), torch.nn.Conv2d(32, 3, 3, 1, 1))
        self.split1 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 1, 1, 0))
        self.split2 = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, 1, 1))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors1 = torch.split(v1, [1, 1, 1], dim=1)
        split_tensors2 = torch.split(v1, [1, 1, 1], dim=1)
        self.save_for_backward(torch.cat(split_tensors1, dim=1))
        return (torch.cat(split_tensors2, dim=1), torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
