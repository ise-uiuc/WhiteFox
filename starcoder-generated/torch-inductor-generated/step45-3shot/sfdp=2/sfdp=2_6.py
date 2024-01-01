
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
    
    def forward(self, inp_query, inp_key, inp_value, inp_key_padding_mask):
        query = inp_query.view(-1, 1, inp_query.size(-1) // self.nhead, self.nhead, self.d_model)
        key = inp_key.view(-1, 1, inp_key.size(-1) // self.nhead, self.nhead, self.d_model)
        value = inp_value.view(-1, 1, inp_value.size(-1) // self.nhead, self.nhead, self.d_model)
        key_padding_mask = inp_key_padding_mask.view(-1, 1, 1, self.nhead)
        v1 = torch.matmul(query, key.transpose(-2, -1)).div(self.d_model ** 0.5)
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=dropout_p)
        v4 = v3.reshape(inp_query.size(-2), inp_query.size(-1), self.nhead, -1).transpose(-2, -1)
        v5 = torch.matmul(v4, value).reshape(v4.size(-1), self.nhead, -1)
        return v5

# Initializing the model
m = Model(d_model=d_model, nhead=nhead)

# Inputs to the model
inp_query = torch.randn(1, 1, 2 * d_model)
inp_key = torch.randn(1, 1, 2 * d_model)
inp_value = torch.randn(1, 1, 2 * d_model)
inp_key_padding_mask = torch.randn(inp_key.size(-2), inp_key.size(-1))
