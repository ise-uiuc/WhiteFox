
class Model(torch.nn.Module):
    def __init__(self, min_value=float("-inf"), max_value=float("inf")):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.FloatTensor([min_value]), requires_grad=False)
        self.variance = torch.nn.Parameter(torch.FloatTensor([max_value]), requires_grad=False)
 
 	def forward(self, x1):
        v1 = torch.nn.Linear(6, 2)
        v2 = v1(x1)
        v3 = torch.clamp_min(v2, min=self.mean)
        v4 = torch.clamp_max(v3, max=self.variance)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
