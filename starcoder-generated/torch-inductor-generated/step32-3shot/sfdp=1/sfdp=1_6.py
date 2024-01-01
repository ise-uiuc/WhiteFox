
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, attn_mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        qk = torch.nn.functional.dropout(qk, self.dropout_p)
        qk = qk.softmax()
        dropout_qk = torch.matmul(value, qk)
        return dropout_qk

# Initializing the model
m = Model(dropout_p=0.5)
# Inputs to the model
query = torch.randn(1, 8, 64)
key = torch.randn(1, 8, 64)
value = torch.randn(1, 8, 64)
attn_mask = torch.randn(1, 1, 64)
# Outputs of the model
