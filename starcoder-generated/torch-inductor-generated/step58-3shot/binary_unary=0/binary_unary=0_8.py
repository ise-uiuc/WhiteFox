
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, 5,stride=1, padding=2),
            torch.nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 16, 3,stride=1, padding=1),
            torch.nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True),
            torch.nn.ReLU(inplace=True)) 
    def forward(self, input_tensor):
        v0 = self.conv_block(input_tensor)
        return v0
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
