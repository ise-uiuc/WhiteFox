
class Model(torch.nn.Module):
	def __init__(self, d):
		super().__init__()
		self.dropout = torch.nn.Dropout(d)
	def forward(self, x):
		x = self.dropout(x)
		return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
