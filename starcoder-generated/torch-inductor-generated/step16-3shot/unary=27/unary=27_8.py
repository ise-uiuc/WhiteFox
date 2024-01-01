
class Model(torch.nn.Module):
    def __init__(self, stride, padding):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(3, 8, 7, stride=stride, padding=padding)
        self.conv1b = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv1a(x)
        x = torch.clamp_min(x, 0.4226538)
        x = torch.clamp_max(x, 0.38730868)
        x = self.conv1b(x)
        x = torch.clamp_min(x, 0.85527596)
        x = torch.clamp_max(x, 0.45974174)
        return x
stride = 1
padding = 0
# Inputs to the model
x = torch.randn(1, 3, 400, 398)
