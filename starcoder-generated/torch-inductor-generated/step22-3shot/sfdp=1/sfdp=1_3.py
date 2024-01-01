
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p=0.5, num_class=10, activation_fn=torch.nn.LeakyReLU(),
                 max_seq_len=100):
        super(Model, self).__init__()
        self.dropout_p = dropout_p

        self.multi_heads = nn.MultiheadAttention(dim, num_heads)
        self.linear1 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x1, x2):
        out, _ = self.multi_heads(x1, x2, x2)
        h = self.linear1(torch.cat([out, x1], dim=-1))
        return h
    
# Initialize the model, with a dimension of 3, and 2 heads
m = Model(3, 2)

# Inputs to the model
x1 = torch.randn(1, 4, 3)
x2 = torch.randn(1, 5, 3)
