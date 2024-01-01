
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features0 = torch.nn.Sequential(torch.nn.BatchNorm2d(8), torch.nn.ReLU6(), torch.nn.MaxPool2d(2, 1, 1))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
