
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.detach().numpy().reshape(-1)
        v3 = v2 > 0
        # v3 is a boolean tensor where each element is True if the corresponding element in v1 is greater than 0, and False otherwise.
        v4 = v2 * 0.1
        v5 = torch.where(v3, v2, v4)
        v6 = torch.from_numpy(v5.reshape(v1.shape)).to(torch.float32)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
