
class Model(torch.nn.Module):
    def __init__(self, min_value=1, max_value=11):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(11, 33, kernel_size=(5, 5), stride=1, padding=0, bias=False)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(33, 22, kernel_size=(1, 1), stride=2, padding=0, bias=True)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        return self.conv2(self.relu(self.conv3(x)))
# Inputs to the model
x = torch.randn(696, 11, 30, 2)
