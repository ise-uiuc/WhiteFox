
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use conv1d instead of linear since kernel size is 1
        self.conv = torch.nn.Conv1d(3, 8, 1, stride=1, padding=0)
        self.bias = torch.nn.Parameter(torch.Tensor(8))
        self.bias.data.fill_(0)
 
    def forward(self, x, other=0):
        v1 = self.conv(x)  # (N, C, L)
        v2 = torch.relu(v1 + self.bias[None, :, None]) 
        v3 = v2 + other
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.zeros(1, 3, 32)
# other is added to the output of the conv() operator.
