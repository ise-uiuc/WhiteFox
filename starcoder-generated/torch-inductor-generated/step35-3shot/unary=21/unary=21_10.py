
class ModelTanh1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 5, 1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(5, 5, 1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(5, 5, 1)
        self.relu3 = torch.nn.ReLU()
    def forward(self, t1, t2):
        x =  self.conv1(t1)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        t3 = torch.tanh(x)
        
        return t3
# Inputs to the model
t1 = torch.randn(1, 2, 32, 32)
t2 = torch.randn(1, 2, 32, 32)
