
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(20, 10)
        self.decoder = torch.nn.Linear(3, 2)

    def forward(self, query, key, value):
        w1 = self.encoder(query)
        w2 = self.encoder(key)
        v = self.decoder(value)

        v1 = torch.matmul(w1, w2.transpose(-2, -1))
        scale_factor = w2.size(-2) ** 0.5
        v2 = v1 * scale_factor
        v3 = torch.softmax(v2, -1)
        dropout_p = 0.0
        v4 = torch.nn.functional.dropout(v3, dropout_p, True, None)
        v5 = torch.matmul(v, v4)

        return v5, v1, v2, v3, v4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 5, 20)
key = torch.randn(2, 6, 20)
value = torch.randn(2, 6, 2)
__output__, __output1__, __output2__, __output3__, __output4__ = m(query, key, value)

