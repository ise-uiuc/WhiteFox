
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_head = MultiHeadAttention(d_model, nhead, batch_first=True, dropout_p=0.1)
        self.value_head = MultiHeadAttention(d_model, nhead, batch_first=True, dropout_p=0.1)
    
    def forward(self, q1, k1, v1):
        q2 = self.query_head(q1, k1)
        v2 = self.value_head(v1, v1)
        return q2, v2

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 2, d_model)
k1 = torch.randn(1, 4, d_model)
v1 = torch.randn(1, 4, d_model)
__output1__, __output2__ = m(q1, k1, v1)

