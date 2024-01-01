
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    def forward(self, x):
        v1 = torch.nn.functional.relu(self.conv1(x))
        v2 = torch.nn.functional.relu(self.conv2(v1))
        v3 = torch.nn.functional.relu(self.conv3(v2))
        
        return v3        
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
