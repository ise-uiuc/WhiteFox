
class Model(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, p=0.1, qk_scale_factor=1.0):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(p)
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.functional.relu
        self.scale_factor = qk_scale_factor
 
    def forward(self, x1):
        x2 = self.embedding(x1)
        x3 = torch.cat(x2)
        x4 = self.linear1(x3)
        x5 = self.activation(x4)
        x6 = self.linear2(x5)
        x7 = self.activation(x6)
        x8 = self.dropout(x7)
        x9 = torch.cat(x8)
        x10 = self.linear1(x9)
        x11 = self.activation(x10)
        return self.linear2(x11)

# Initializing the model
m = Model(vocab_size=100000, hidden_size=64, p=0.2, qk_scale_factor=0.5)

# Inputs to the model
x1 = torch.randint(5, (6, 16))
