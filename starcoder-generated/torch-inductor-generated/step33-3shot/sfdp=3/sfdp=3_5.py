
torch.set_printoptions(threshold=2000)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 2
        hidden_size = 2
        num_heads = 1
        dropout_p = 0.1
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linin2 = torch.nn.Linear(hidden_size, num_heads)
        self.linear3 = torch.nn.Linear(hidden_size, 1)
        self.scale_factor = math.sqrt(hidden_size) / math.pow(num_heads, 0.5)
 
    def forward(self, query, key, value):
        k = self.linear1(key)
        q = self.linear2(query)
        v = self.linear3(value)
        qkp = torch.matmul(q, k.transpose(-2, -1))
        scaled_qkp = qkp.mul(self.scale_factor)
        softmax_qkp = scaled_qkp.softmax(dim=-1)
        dropout_qkp = torch.nn.functional.dropout(softmax_qkp, p=dropout_p)
        output = dropout_qkp.matmul(v)
        return output

# Initializing the model
x1 = torch.randn(1, 1, 2)
x2 = torch.randn(1, 1, 2)
m = Model()

# Input to the model
