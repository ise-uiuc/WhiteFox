
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = torch.nn.Linear(8, 3, bias=True)
        self.linear_k = torch.nn.Linear(5, 8, bias=True)
        self.linear_v = torch.nn.Linear(5, 3, bias=True)
        self.scale_factor = 1.0
        self.dropout_p = 0.5

    def forward(self, q, k, v):
        k, v = self.linear_k(k), self.linear_v(v)
        query = self.linear_q(q)
        qk = torch.matmul(query, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 5, 8) # Shape: [Sequence length, Channel size]
k = torch.randn(1, 3, 8) # Shape: [Sequence length, Channel size]
v = torch.randn(1, 3, 3) # Shape: [Sequence length, Channel size]
