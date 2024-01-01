
class Model(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p=0.5, inv_scale_factor=0.5):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        outputs = output, dropout_qk
        return outputs
 
# Initializing the model
q = torch.randn(1, 4, 128)
k = torch.randn(1, 4, 128)
v = torch.randn(1, 4, 128)
model = Model(q, k, v, 0.5, 0.5)

# Inputs to the model
