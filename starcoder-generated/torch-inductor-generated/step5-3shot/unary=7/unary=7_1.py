
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
 
        self.linear = torch.nn.Linear(64 * 51 * 51, 512)       
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(torch.rand(512, 64 * 51 * 51))
    
    def forward(self, x1):
        l1 = self.linear(self.conv(x1).view(-1))
        l2 = l1 * torch.clamp(l1.max(), 0, 6) + 3
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
