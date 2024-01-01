
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 3, 2, 3), torch.nn.Conv2d(32, 32, 3, 1, 1))
        self.concat = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 3, 3), torch.nn.Conv2d(6, 1, 3, 1, 1))
    def forward(self, v1):
        split_tensors = torch.split(v1, 2, dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, 2, dim=1))
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
