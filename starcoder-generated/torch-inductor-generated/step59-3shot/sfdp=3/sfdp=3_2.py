
class Model(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.scale_factor = torch.sqrt(torch.FloatTensor([n_embd])).to(get_device())
    
    def forward(self, x):
        query = torch.rand(x, x, x, dtype=torch.float32, requires_grad=True, device=get_device())
        key = torch.rand(x, x, x, dtype=torch.float32, requires_grad=True, device=get_device())
        value = torch.rand(x, x, x, dtype=torch.float32, requires_grad=True, device=get_device())

        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2,training=True)
        output = dropout_qk.matmul(value)

        return output

# Initializing the model
m = Model(1024)

# Inputs to the model
x = 1
y = torch.randn(x,x,x, dtype=torch.float32, requires_grad=True, device=get_device())
