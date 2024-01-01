
class Model(torch.nn.Module):
    def __init__(self, head_num, hidden_dim, input_dim, value_dim, query_dim, dropout_p):
        super(Model, self).__init__()
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.value_dim = value_dim
        self.query_dim = query_dim
        self.dropout_p = dropout_p
 
        self.W_Query = torch.nn.Parameter(torch.rand((head_num, query_dim, hidden_dim // head_num)))
        self.W_Key = torch.nn.Parameter(torch.rand((head_num, input_dim, hidden_dim // head_num)))
        self.W_Value = torch.nn.Parameter(torch.rand((head_num, value_dim, hidden_dim // head_num)))
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_p=dropout_p)
 
    def forward(self, _input, hidden):
        query = torch.matmul(_input, self.W_Query)
        key = torch.matmul(hidden, self.W_Key)
        value = torch.matmul(hidden, self.W_Value)
        context, attentions = self.scaled_dot_product_attention(query, key, value)
        return context, attentions

# Initializing the model
m = Model(8, 16, 32, 128, 20, 0.3)

# Inputs to the model
input_ = torch.rand(128, 32)
hidden = torch.rand(128, 16)
__output__, __attention__ = m(input_, hidden)

