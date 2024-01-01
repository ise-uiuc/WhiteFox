
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.2
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        self.div = torch.div
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 30, 4)
key = torch.randn(1, 40, 4)
value = torch.randn(1, 40, 8)
inv_scale_factor = 0.2
