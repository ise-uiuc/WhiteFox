
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.detach().clone()
        zeros = torch.zeros_like(v1)
        mask = v1 > 0
        neg_part = v2 * -0.01
        result = torch.where(mask, v2, neg_part)
        return result

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8)
