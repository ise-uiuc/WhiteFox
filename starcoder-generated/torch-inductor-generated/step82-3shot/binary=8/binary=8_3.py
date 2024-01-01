
class Model(torch.nn.Module):
    def __init__(self):
        import numpy as np        
        super().__init__()
        np.random.seed  (10)
        torch.manual_seed(10)
        self.conv1 = torch.nn.Conv2d(1, 4, 5, padding=0, stride=1)
    def forward(self, x):
        x1 = x.view(-1, 1, 28, 28)
        x2 = self.conv1(x1)
        x3 = x2.view(-1, 10)
        # x4 = torch.tensor(0.5)
        x4 = torch.randn(1)
        return x4
# Inputs to the model
torch.manual_seed(20)
np.random.seed(20)
x = torch.randn(5, 1, 28, 28)
