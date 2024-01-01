
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 7, 1, 3), torch.nn.ReLU(), torch.nn.Conv2d(32, 3, 7, 1, 3), torch.nn.ReLU())
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 2], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 2], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
