
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.1, query=None, key=None, value=None):
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax = torch.nn.Softmax(dim=-1)
        if query is None:
            query, key, value= generate_model_input('attention', [1, 128], [1, 128], [1, 128])
        self.register_buffer('query', query)
        self.register_buffer('key', key)
        self.register_buffer('value', key)
 
    def forward(self, x):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.query.size(-1)**-0.5)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
