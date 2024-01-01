
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.rand(128, 16))
        self.val = torch.nn.Parameter(torch.rand(128, 16))
        self.dropout_p = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32))
 
    def forward(self, query):
        k = self.key
        v = self.val
        p = self.dropout_p
        inv_scale = 10
        qk = torch.matmul(query, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(32, 128, 16)
