
class Model(torch.nn.Module):
    def __init__(self, dropout_p, inv_scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query, key, value):
        qk = query.matmul(key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(0.1, 0.5)

# Input to the model
query = torch.randn(16, 32, 512)
key = torch.randn(16, 32, 256)
value = torch.randn(16, 32, 256)

