
class Model(torch.nn.Module):
    def __init__(self, min_clamp=0.7, max_clamp=0.6):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 4, 1, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride=1, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(4, 8, 3, stride=1, padding=2, bias=True)
        self.min_clamp = torch.zeros(8) + min_clamp
        self.max_clamp = torch.zeros(8) + max_clamp
    def forward(self, x1):
        v1 = torch.nn.functional.relu(self.conv1(x1))
        v2 = torch.nn.functional.relu(self.conv2(v1))
        v3 = torch.nn.functional.relu(self.conv3(v1))
        v4 = self.min_clamp.reshape(1, 8, 1, 1)
        return F.relu(torch.clamp(v2 * v3 + v4, min=self.min_clamp, max=self.max_clamp))
# Inputs to the model
x1 = torch.randn(2, 5, 6, 6)
