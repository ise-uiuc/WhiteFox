
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        query = self.ln1(query)
        key = self.ln2(key)
        value = self.ln3(value)
        qk = torch.matmul(query, key.transpose(1,2))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=2)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        output = self.ln4(output)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 128, 32, 32)
key = torch.randn(1, 128, 32, 32)
value = torch.randn(1, 128, 32, 32)
inv_scale_factor = torch.ones([1, 1])
dropout_p = 0.2
