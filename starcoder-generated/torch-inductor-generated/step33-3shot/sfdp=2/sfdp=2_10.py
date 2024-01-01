
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Q, K, V, seq_len, attention_mask=None, dropout_p=0.0):
        QK = torch.matmul(Q, K.transpose(-2, -1))
        inv_scale_factor = seq_len**(-0.5)
        scaled_QK = QK.div(inv_scale_factor)
        softmax_QK = scaled_QK.softmax(dim=-1)
        if dropout_p > 0:
            dropout_QK = torch.nn.functional.dropout(softmax_QK, p=dropout_p)
        else:
            dropout_QK = softmax_QK
        output = dropout_QK.matmul(V)
        return output

# Initializing the model
m = Model()

# Inputs to the model
Q = torch.randn(2, 4, 5)
K = torch.randn(2, 5, 6)
V = torch.randn(2, 5, 6)
seq_len = 6
attention_mask = (torch.tril(torch.ones(2, 6, 6)) == 0).unsqueeze(1).unsqueeze(1) # A tensor which matches with the shape of the query
dropout_p = 0.1
