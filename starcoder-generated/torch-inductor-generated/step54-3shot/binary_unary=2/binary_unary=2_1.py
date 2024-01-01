
class Model(torch.nn.Module):
    def __init__(self):
	super().__init__()
	self.conv1 = torch.nn.Conv2d(8, 8, 1)
	self.conv2 = torch.nn.Conv2d(8, 8, 1)
    def forward(self, X0):
	v1 = self.conv1(X0)
	v2 = self.conv2(X0)
	v3 = v1 - v2
        v4 = F.relu(v3)
        return v4
# Inputs to the model
X0 = torch.randn(1, 8, 64, 64)
