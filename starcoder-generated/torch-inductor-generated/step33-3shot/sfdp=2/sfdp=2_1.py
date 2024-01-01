
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.query = torch.nn.Parameter(torch.Tensor([0.7152, 0.0001, 0.4840, 0.9726, 0.2713, 0.4230, 0.1675, 0.2349]), requires_grad=True)
        self.key = torch.nn.Parameter(torch.Tensor([0.8767, 0.8070, 0.1362, 0.0070, 0.5368, 0.5706, 0.5209, 0.7151]), requires_grad=True)
        self.value = torch.nn.Parameter(torch.Tensor([0.2789, 0.3270, 0.1970, 0.6310, 0.9674, 0.1790, 0.5554, 0.8951]), requires_grad=True)
        self.inv_scale_factor = 1 / math.sqrt(key_size)
 
    def forward(self, x1):
        q = self.query
        k = self.key
        v = self.value
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(8, 8, 8, 0.01479775)

# Inputs to the model
x1 = torch.randn(1, 8, 15)
