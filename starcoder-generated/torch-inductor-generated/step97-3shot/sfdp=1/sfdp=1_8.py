
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_tensor):
        v1 = self.linear1(input_tensor)
        v2 = v1.div(0.1)
        v3 = v1.div(1.1)
        v4 = torch.matmul(v2, v3.transpose(0, 1))
        v5 = v4.softmax(dim=-1)
        v6 = torch.nn.functional.dropout(v5, 0.5)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(batch_size, text_encoder_seq_lens, hidden_size)
