
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(1, 3, 3, 1, 1, bias=False), torch.nn.Conv2d(3, 1, 3, 1, 1, bias=False), torch.nn.Conv2d(1, 3, 5, 1, 1, bias=False), torch.nn.Conv2d(3, 1, 3, 1, 1, bias=False))

    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor)
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
