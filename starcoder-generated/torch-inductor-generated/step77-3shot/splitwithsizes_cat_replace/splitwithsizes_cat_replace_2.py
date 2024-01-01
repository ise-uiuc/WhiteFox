
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = [torch.nn.Conv2d(3, 32, 3, stride=2, padding=1, dilation=1, bias=False)]
        self.features_1 = [torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.03)]
        self.features_2 = [torch.nn.ReLU6()]
        self.features_3 = [torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1, bias=False)]
        self.features_4 = [torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)]
        self.features_5 = [torch.nn.ReLU6()]
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return torch.tanh(concatenated_tensor)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 14)
