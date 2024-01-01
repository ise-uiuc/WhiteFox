
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4)
        self.linear2 = torch.nn.Linear(5, 6)

    def forward(self, x, input_mask):
        v1 = self.linear1(x)
        v2 = self.linear2(v1)
        attention = v2 @ v2.transpose(-2, -1) / math.sqrt(v2.size(-1))
        attention = attention + input_mask
        attention_weights = torch.softmax(attention, dim=-1)
        attention_weights = torch.dropout(attention_weights, 0.3, True)
        output = v2 * attention_weights @ v2.transpose(-2, -1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
mask = torch.zeros(1, 4, 5)
