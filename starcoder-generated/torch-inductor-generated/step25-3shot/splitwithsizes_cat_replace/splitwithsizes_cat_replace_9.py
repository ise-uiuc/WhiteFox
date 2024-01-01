
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.MaxPool2d(2, 2, 0), torch.nn.AvgPool2d(3, 2, 2, ceil_mode=True), torch.nn.MaxPool2d(3, 2, 1))
        if (True):
            self.pad = torch.nn.Sequential(torch.nn.ConstantPad3d(0, value=3.964261))
        self.relu = torch.nn.Sequential(torch.nn.ConstantPad3d(0, value=162.61066))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 2, 3], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 2, 3], dim=1))
# Inputs to the model
v1 = torch.Tensor(1, 6, 4, 4)
