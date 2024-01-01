
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 32, bias=False)
 
    def forward(self, x1, x2):
        q = self.linear(x1)
        k = self.linear(x1)
        scale_factor = np.sqrt(q.shape[-1])
        v = self.linear(x2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 128)
x2 = torch.randn(16, 64)
