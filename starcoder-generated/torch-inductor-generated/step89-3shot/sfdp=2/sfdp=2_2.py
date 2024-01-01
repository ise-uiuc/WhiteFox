
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, value, inv_scale_factor, dropout_p):
        qk = query @ key.transpose(-2, -1)
        scaled_qk = qk / inv_scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 28, 48)
value = torch.randn(1, 8, 28, 48)
inv_scale_factor = 1e-6
dropout_p = 0.5
