
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features0 = torch.nn.Sequential(torch.nn.MaxPool2d(3, 2, 0), torch.nn.Conv2d(3, 64, 1, 1, 0, bias=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
