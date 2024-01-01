
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 32, 5, 1, 2), torch.nn.ReLU(), torch.nn.ConvTranspose2d(-1, 32, 3, 1, 0), torch.nn.ReLU(), torch.nn.ConvTranspose2d(32, 32, 3, 1, 1), torch.nn.ReLU(), torch.nn.ConvTranspose2d(32, 3, 3, 1, 1), torch.nn.Sigmoid())
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
