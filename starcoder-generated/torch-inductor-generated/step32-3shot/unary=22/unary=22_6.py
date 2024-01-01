
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.norm = torch.nn.BatchNorm2d(num_features=8, track_running_stats=True)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.norm(v1)
        v3 = self.norm(v2)
        v4 = self.norm(v3)
        w_tanh = torch.nn.Tanh(v4)
        return w_tanh

# Initializing and generating dummy data
inputData = torch.randn(2, 4, 16, 16)
norm = torch.nn.BatchNorm2d(num_features=4, track_running_stats=True)
m = Model()

# Inputs to the model
