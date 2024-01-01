
class Model(torch.nn.Module):
    def __init__(self,
                 query_size,
                 key_size,
                 value_size,
                 scale_factor=1, # Default value is 1
                 droput_p=0.1): # Default value is 0.1
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = droput_p
        self.query = torch.nn.Linear(query_size, key_size, bias=False)
        self.key = torch.nn.Linear(key_size, query_size, bias=False)
        self.value = torch.nn.Linear(value_size, query_size)

    def forward(self, x1, x2, x3, dropout_p=None):
        if dropout_p == None:
            dropout_p = self.dropout_p
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x3)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output, softmax_qk

# Initializing the model
m = Model(query_size=13,
          key_size=17,
          value_size=23)

# Inputs to the model
x1 = torch.randn(5, 13)
x2 = torch.randn(10, 17)
x3 = torch.randn(13, 23)
__output__, __attention__ = m(x1=x1, x2=x2, x3=x3, dropout_p=0.3)
