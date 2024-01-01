
class B(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = torch.nn.Conv2d(3, 32, kernel_size=(1,3), stride=1, padding=(0,1))
	def forward(self, x1):
		v1 = self.conv(x1)
		return v1
# Input to the model
x1 = torch.tensor(1.0)
