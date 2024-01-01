
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch expects the first dimension of the input tensor to be the batch dimension. Since the batch dimension can vary at inference time (i.e. it depends on the number of images fed into the model at a time), the batch dimension should be set to a placeholder value of 1 instead of the true batch dimension at model construction time.
        self.linear = torch.nn.Linear(64, 1, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1).squeeze(dim=1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
