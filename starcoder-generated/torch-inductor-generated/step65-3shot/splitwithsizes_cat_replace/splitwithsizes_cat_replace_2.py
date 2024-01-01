
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(*(torch.nn.MaxPool2d(3, 2, 1) for _ in range(21)))
    def forward(self, v1):
        split_tensors = torch.split(v1, [4, 4, 4], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [4, 4, 4], dim=1))
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
