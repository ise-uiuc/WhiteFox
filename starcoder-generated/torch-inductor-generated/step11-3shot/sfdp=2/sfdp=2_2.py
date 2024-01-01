
class Model(torch.nn.Module):
    def __init__(self, hidden_dim, dropout_p):
        super().__init__()
        self.scale_factor = np.power(hidden_dim, 0.5)
        self.inv_scale_factor = 1 / np.power(hidden_dim, 0.5)
 
    def forward(self, query, key, value, dropout_p):
        QK = torch.matmul(query, key.transpose(-2, -1))
        scaled_QK = QK.div(self.inv_scale_factor)
        softmax_QK = scaled_QK.softmax(dim=-1)
        dropout_QK = torch.nn.functional.dropout(softmax_QK, p=dropout_p)
        output = dropout_QK.matmul(value)
        return output

# Initializing the model
m = Model(32, 0.1)

# Inputs to the model
x1 = torch.randn(1, 16, 32)
x2 = torch.randn(1, 16, 32)
x3 = torch.randn(1, 16, 32)
