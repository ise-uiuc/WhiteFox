
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = F.avg_pool2d(v2, 10, stride=4) 
        v4 = torch.nn.functional.linear(v3.flatten(start_dim=1), torch.tensor([[0.1, 0.2, -0.1]]))
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
