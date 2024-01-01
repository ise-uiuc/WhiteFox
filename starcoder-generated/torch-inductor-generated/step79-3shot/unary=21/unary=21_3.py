
class ModelTanh(torch.nn.Module):
	def __init__(self):
		super(ModelTanh, self).__init__()
		self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=(3, 5), stride=(4, 5))
		self.conv2 = torch.nn.ConvTranspose1d(4, 8, kernel_size=3, stride=5)
		self.conv3 = torch.nn.ConvTranspose1d(8, 1, kernel_size=3, stride=1)
		self.conv4 = torch.nn.Conv1d(1, 9, kernel_size=3)
		self.conv5 = torch.nn.Conv1d(9, 4, kernel_size=(3, 5), stride=(1, 2))
	def forward(self, x):
		x = self.conv1(x)
		x = torch.tanh(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = torch.tanh(x)
		x = self.conv4(x)
		x = torch.tanh(x)
		x = self.conv5(x)
		x = torch.tanh(x)
		return x
# Inputs to the model
x1 = torch.randn(1, 1, 58)
