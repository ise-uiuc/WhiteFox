
class Model(torch.nn.Module):
    def forward(self, query, key, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1)).div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return qk, softmax_qk, dropout_qk, output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(32, 8, 1024)
key = torch.randn(64, 8, 1024)
inv_scale_factor = torch.tensor(10.0)
dropout_p = torch.tensor(0.1)
qk, softmax_qk, dropout_qk, output = m(query, key, inv_scale_factor, dropout_p)
