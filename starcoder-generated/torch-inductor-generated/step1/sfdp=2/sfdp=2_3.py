
class Model(torch.nn.Module):
    def __init__(self, dropout_p, num_heads, d_k, dropout_q, dropout_v):
        super().__init__()
        # Initializers are randomly generated
        self.q = torch.nn.Linear(512, 512, bias=False)
        self.k = torch.nn.Linear(512, 512, bias=False)
        self.v = torch.nn.Linear(512, 512, bias=False)
        self.dropout_p = dropout_p
        self.drop = torch.nn.Dropout(dropout_p)
 
    def forward(self, x):
        temp1 = self.q(x)
        temp2 = self.k(x)
        temp3 = torch.softmax(temp1.bmm(temp2.transpose(-2, -1))/math.sqrt(512), -1)
        temp4 = self.drop(temp3)
        temp5 = self.v(x)
        return temp4.bmm(temp5)

# Initializing the model
m = Model(0.4, 4, 16, 0.1, 0.2)

# Inputs to the model
x = torch.randn(1, 512)
