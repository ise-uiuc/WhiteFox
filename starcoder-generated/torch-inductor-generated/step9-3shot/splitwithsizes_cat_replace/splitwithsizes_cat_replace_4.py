
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(120, 96, 3, 1, 1), torch.nn.Conv2d(96, 120, 3, 1, 1))
        self.split = torch.nn.Conv2d(in_channels=120, out_channels=3, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(self.split(v1), 3, dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, 3, dim=1))
# Inputs to the model
x1 = torch.randn(1, 120, 16, 16)
