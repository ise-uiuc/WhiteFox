
class clamp_model(nn.Module):
    def __init__(self, min_value=0.0, max_value=0.01):
        super(clamp_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=0)
        #self.conv2 = nn.Conv2d(25,84,kernel_size=5, stride=1,padding=0)
        self.conv3 = nn.Conv2d(32, 84, kernel_size=5, stride=2, padding=1, groups=32)
        self.relu = nn.ReLU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv3(v3)
        output = self.relu(v4)
        return output
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
