
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(2, 200, 10))
        self.key = torch.nn.Parameter(torch.randn(2, 200, 20))
        self.value = torch.nn.Parameter(torch.randn(2, 10, 20))
 
    def forward(self, x):
        q = self.query
        k = self.key
        v = self.value
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = np.sum(q**2 + k**2, 2).sqrt().reciprocal().unsqueeze(-1)
        softmax_qk = scaled_qk.div(inv_scale_factor)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.25)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 15, 200)
y = m(x)

