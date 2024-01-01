
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1 / self.dropout_p
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
inv_scale_factor = random.random()
dropout_p = random.random()
m = Model(inv_scale_factor, dropout_p)
print("Dropout probability: " + str(dropout_p))
 
# Inputs to the model
query = torch.randn(1, 64, 100)
key = torch.randn(1, 100, 200)
value = torch.randn(1, 200, 300)
output = m(query, key, value)

