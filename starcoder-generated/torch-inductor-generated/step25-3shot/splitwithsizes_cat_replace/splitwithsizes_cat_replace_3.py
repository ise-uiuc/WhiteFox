
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.MaxPool2d(2, stride=32, padding=(0, 2)), torch.nn.MaxPool2d(2, stride=16, padding=(0, 1)), torch.nn.AvgPool2d(3, stride=8, padding=(0, 0)))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
