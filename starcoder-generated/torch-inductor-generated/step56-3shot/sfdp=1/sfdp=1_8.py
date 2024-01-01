
class Model(torch.nn.Module):
    def __init__(self, attention_heads, dropout_p, d_model):
        super().__init__()
        self.dropout_p = dropout_p
        self.attention_heads = attention_heads
        self.d_model = d_model
        assert self.d_model % self.attention_heads == 0
        self.d_head = int(d_model / self.attention_heads)
        self.query = torch.nn.Linear(self.d_model, self.d_model)
        self.key = torch.nn.Linear(self.d_model, self.d_model)
        self.value = torch.nn.Linear(self.d_model, self.d_model)
        self.combine = torch.nn.Linear(self.d_model, self.d_model)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
 
    def forward(self, x1, x2):
        query = self.dropout(self.softmax(torch.matmul(self.query(x1), torch.transpose(self.key(x2), -2, -1)).div(math.sqrt(self.d_model))))
        output = torch.matmul(self.value(x2), torch.transpose(query, -2, -1))
        output = self.combine(output)
        return output

# Initializing the model
m = Model(attention_heads=4, dropout_p=0.8, d_model=128)

# Inputs to the model
x1 = torch.randn(2, 10, 128)
x2 = torch.randn(2, 20, 128)
