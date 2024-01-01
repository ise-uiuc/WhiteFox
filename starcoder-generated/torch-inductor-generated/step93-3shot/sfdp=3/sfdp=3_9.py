
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor, dropout_p):
        super().__init__()
        self.weight_query = torch.nn.Parameter(torch.FloatTensor())
        self.weight_key = torch.nn.Parameter(torch.FloatTensor())
        self.weight_value = torch.nn.Parameter(torch.FloatTensor())
        torch.nn.init.normal_(self.weight_query)
        torch.nn.init.normal_(self.weight_key)
        torch.nn.init.normal_(self.weight_value)
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
q = torch.randn(1, 16, 32)
k = torch.randn(1, 16, 32)
v = torch.randn(1, 16, 32)
sf = torch.randn(1, 16, 16)
dp = 0.2
