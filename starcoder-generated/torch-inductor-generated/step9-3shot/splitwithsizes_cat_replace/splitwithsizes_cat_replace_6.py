
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0), torch.nn.ReLU6(inplace=True))
        self.split = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1, dilation=1, ceil_mode=(0, 0)), torch.nn.MaxPool2d(kernel_size=(5, 5), stride=4, padding=2, dilation=1, ceil_mode=(0, 0)), torch.nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=0, dilation=1, ceil_mode=(0, 0)))
    def forward(self, x4):
        v4 = self.features(x4)
        split_tensors = torch.split(v4, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
# Inputs to the model
x4 = torch.randn(1, 3, 64, 64)
