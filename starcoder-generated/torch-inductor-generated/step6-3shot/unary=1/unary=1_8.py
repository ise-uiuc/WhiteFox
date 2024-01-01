ing
class Model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.add = torch.nn.Add()
		self.mul = torch.nn.Mul()
		self.tanh = torch.nn.Tanh()
		self.mul1 = torch.nn.Mul()
		self.add_scalar = torch.nn.AddScalar()

	# forward defines the computation performed at every call.
	def forward(self, x1, x2, x3, x4):
		v1 = self.add(x1, x2)
		v2 = self.mul(x3, x4)
		v3 = self.add(x3, v2)
		v4 = self.tanh(x4)
		v5 = self.mul(v3, v4)
		v6 = self.add_scalar(v5, 1)
		v7 = self.mul(v1, v6)
		return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
