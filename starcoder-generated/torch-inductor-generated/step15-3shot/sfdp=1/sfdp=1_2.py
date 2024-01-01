
class Model(torch.nn.Module):
    def __init__(self, num_heads, num_qk, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.num_qk = num_qk
 
        self.W_query = torch.nn.Parameter(torch.FloatTensor(num_heads, num_qk, 32))
        self.W_key = torch.nn.Parameter(torch.FloatTensor(num_heads, num_qk, 32))
        self.W_value = torch.nn.Parameter(torch.FloatTensor(num_heads, num_qk, 32))
 
        self.dropout_p = dropout_p
 
        self.scaled_qk = None
 
    def forward(self, x1, x2):
        qb = torch.bmm(x1, self.W_query)
        kb = torch.bmm(x2, self.W_key)
        self.scaled_qk = torch.bmm(qb.transpose(1, 2), kb)
 
        num_units = self.num_heads * self.num_qk
 
        scaled_qk = self.scaled_qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.W_value.reshape(num_units, -1).transpose(1, 0).unsqueeze(0))
 
        output = output.reshape(x1.shape[0], -1, output.shape[-1]).transpose(1, 2)
        return output

# Initializing the model
m = Model(num_heads=8, num_qk=64, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(1, 32, 64)
x2 = torch.randn(1, 32, 64)
__output1__, __output2__ = m(x1, x2)

# Inputs to the model
x1 = torch.randn(1, 32, 64)
x2 = torch.randn(1, 32, 64)
__output3__, __output4__ = m(x1, x2)