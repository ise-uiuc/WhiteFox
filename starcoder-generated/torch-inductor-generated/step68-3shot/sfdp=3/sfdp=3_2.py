
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(3, 8, 19, 16))
        self.key = torch.nn.Parameter(torch.randn(3, 4, 23, 9))
        self.scale_factor = 4.0
        self.dropout_p = 0.2
 
    def forward(self, q):
        query = self.query
        key = self.key
        scale_factor = self.scale_factor
        dropout_p = self.dropout_p
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        v = torch.matmul(dropout_qk, value)
        return v

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 3, 22, 5)
v = torch.randn(1, 3, 93, 6)
