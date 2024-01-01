
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = float(1 / np.sqrt(hidden_size))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, q_length, hidden_size)
k = torch.randn(1, k_length, hidden_size)
