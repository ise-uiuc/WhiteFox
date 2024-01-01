
class MyAttention(torch.nn.Module):
    def __init__(self, input_size, head_num, dropout_p):
        super().__init__()
 
        self.input_size = input_size
        self.head_num = head_num
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(dropout_p)
 
        self.Q_linear = torch.nn.Linear(input_size, input_size)
        self.K_linear = torch.nn.Linear(input_size, input_size)
        self.V_linear = torch.nn.Linear(input_size, input_size)
 
        self.output_linear = torch.nn.Linear(input_size, input_size)
 
    def forward(self, q, k, v, inv_scale_factor=1):
        residual = q
        q = self.Q_linear(q)
        k = self.K_linear(k)
        v = self.V_linear(v)
 
        q = q.view(q.size(0), q.size(1), self.head_num, q.size(2)//self.head_num)
        k = k.view(k.size(0), k.size(1), self.head_num, k.size(2)//self.head_num)
        v = v.view(v.size(0), v.size(1), self.head_num, v.size(2)//self.head_num)
 
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
 
        qk = torch.matmul(q, k)
        qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        softmax_qk = self.dropout(softmax_qk)
 
        dropout_qk = torch.matmul(softmax_qk, v)
        dropout_qk = dropout_qk.transpose(-2, -1).contiguous()
        dropout_qk = dropout_qk.view(dropout_qk.size(0), dropout_qk.size(1), dropout_qk.size(2)*dropout_qk.size(3))
 
        output = self.output_linear(dropout_qk)
        output += residual
 
        return output, softmax_qk
 
class Model(torch.nn.Module):
    def __init__(self, input_size, head_num, dropout_p):
        super().__init__()
 
        self.attention = MyAttention(input_size, head_num, dropout_p)
 
    def forward(self, x, inv_scale_factor=10):
        attn1, attn2 = self.attention(x, x, x, inv_scale_factor)
        output = attn1 * attn2
        return output

# Initializing the model
m = Model(1024, 4, 0.2)
 
# Inputs to the model
x = torch.randn(1, 8, 1024)
