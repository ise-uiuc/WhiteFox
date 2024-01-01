
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.inv_scale_factor = np.sqrt(self.dropout_p)
 
    def forward(self, *xs):
        query = xs[0]
        key = xs[1]
        value = xs[2]
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 64)
key = torch.randn(16, 64)
value = torch.randn(16, 64)
