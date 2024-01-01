
class Model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(8, 4)
		self.f2 = nn.Linear(4, 8)
		
	def forward(self, x):
		n1 = self.fc1(x)
		n2 = self.f2(n1)
		output = n1 * n2
		return output
   
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
