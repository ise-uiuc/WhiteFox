
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, 1, 1), torch.nn.Conv2d(16, 32, 7, 2, 2), torch.nn.Conv2d(32, 16, 5, 1, 1), torch.nn.Conv2d(16, 8, 13, 2, 2), torch.nn.ConvTranspose2d(8, 1, 7, 1, 0), torch.nn.ConvTranspose2d(8, 16, 4, 1, 0))
        self.split = torch.nn.Sequential(torch.nn.MaxPool2d(5, 3, 0, 1), torch.nn.MaxPool2d(4, 2, 2, 2), torch.nn.MaxPool2d(3, 1, 2, 1))
    def forward(self, x1):
        v1 = self.features(x1)
        split_tensors = torch.split(v1, [9, 5, 9], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [9, 5, 9], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
