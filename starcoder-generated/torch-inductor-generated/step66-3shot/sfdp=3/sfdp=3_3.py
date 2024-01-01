
class Model(torch.nn.Module):
    def __init__(self, query, key, value):
        super().__init__()
        self.register_buffer('query', query)
        self.register_buffer('key', key)
        self.register_buffer('value', value)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.5)
 
    def forward(self, scale_factor=None, dropout_p=None):
        if (scale_factor is None):
            scale_factor = 1.0
        if (dropout_p is None):
            dropout_p = 0
        scaled_qk = torch.matmul(self.query, self.key.transpose(-2, -1)) * scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, self.value)
        return output

# Initializing the model
m = Model(query, key, value)

# Inputs to the model
