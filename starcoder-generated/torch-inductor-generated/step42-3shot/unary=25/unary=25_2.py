
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=2, out_features=4)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
 
        return v4

# Initializing the model
m = Model()
print("Positive slope:", m(torch.tensor([1, 1])).tolist())
print("Negative slope:", m(torch.tensor([1, -1])).tolist())

