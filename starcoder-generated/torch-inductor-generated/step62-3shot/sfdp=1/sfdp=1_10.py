
class Model(torch.nn.Module):
    def __init__(self, shape):
        super(Model, self).__init__()
        self.shape = shape
        self.q = torch.nn.Parameter(torch.randn(*shape))
        self.v = torch.nn.Parameter(torch.randn(*shape))
        self.k = torch.nn.Parameter(torch.randn(*shape))
        
    def forward(self, x1, x2):
        softmax_qk = torch.matmul(self.q, self.k.transpose(-2, -1))
        softmax_qk = softmax_qk.softmax(dim=1)
        x = softmax_qk.matmul(self.v)
        return x

# Initializing the model
m = Model(shape=(5, 4, 6))

# Inputs to the model
x1 = torch.randn(1, 5, 4, 6)
x2 = torch.randn(1, 4, 5, 6)
