
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p=0.5):
        super().__init__()
        self.dropout_p = dropout_p
        self.q = torch.nn.Parameter(torch.rand(2, dim, requires_grad=True))
        self.k = torch.nn.Parameter(torch.rand(2, dim, requires_grad=True))
        self.v = torch.nn.Parameter(torch.rand(2, dim, requires_grad=True))
 
    def forward(self, inputs):
        w = torch.matmul(inputs, self.q.transpose(-2, -1))
        w = w / math.sqrt(self.q.shape[-1])
        w = torch.nn.functional.softmax(w, dim=-1)
        w = torch.nn.functional.dropout(w, p=self.dropout_p)
        z = torch.matmul(w, self.v)
        return z

# Initializing the model
m = Model(3, 0.9)

# Inputs to the model
inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
