
class Model(torch.nn.Module): 
    def __init__(self, hidden_dim, num_heads, attention_dropout, use_bias): 
        super().__init__() 
        self.query_projection = torch.nn.Linear(hidden_dim, hidden_dim, bias=use_bias) 
        self.key_projection = torch.nn.Linear(hidden_dim, hidden_dim, bias=use_bias) 
        self.value_projection = torch.nn.Linear(hidden_dim, hidden_dim, bias=use_bias) 
 
        self.dropout = torch.nn.Dropout(attention_dropout, inplace=False) 
 
        self.num_heads = num_heads 
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p): 
        query, key, value = self.query_projection(query), self.key_projection(key), self.value_projection(value) 
        keydim = key.dtype 
        inv_scale_factor = inv_scale_factor.to(keydim) 
        dropout_p = dropout_p ** self.num_heads # Scale the dropout probability by the number of attention heads 
 
        scaled_qk, orig_q, orig_k = ScaledDotProductAttention()(query, key, value, query, 0, 1) 
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = self.dropout(softmax_qk) 
        output = torch.matmul(dropout_qk, value) 
        return output
 
# Initializing the model 
m = Model(hidden_dim=32, num_heads=8, attention_dropout=0.1, use_bias=True) 
 
# Inputs to the model 
query = torch.randn(1, 2, 32) 
key = torch.randn(1, 2, 32) 
value = torch.randn(1, 8, 32) 
inv_scale_factor = torch.randn(1, 8, 8, 8) 
dropout_p = torch.tensor([0.00]) 
 
