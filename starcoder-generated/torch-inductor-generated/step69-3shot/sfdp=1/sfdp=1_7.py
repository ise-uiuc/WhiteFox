
class Model(torch.nn.Module):
    def __init__(self, dim, input_shape):
        super().__init__()
        self.dim = dim
        self.inv_scale_factor = 1 / np.sqrt(dim)
        self.dropout_p = 0
        self.qk = torch.nn.Linear(input_shape, dim, bias=False)
        self.value = torch.nn.Linear(input_shape, dim, bias=False)

    def forward(self, x1):
        v0 = self.qk(x1)
        v1 = v0.div(self.inv_scale_factor)
        v2 = torch.nn.functional.dropout(v1.softmax(dim=-1), p=self.dropout_p)
        v3 = self.value(x1)
        v4 = v2.matmul(v3) 
        return v4

# Initializing the model
m = Model(dim=32, input_shape=128)

# Inputs to the model
x1 = torch.randn(1, 128)
