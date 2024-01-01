
class Model(torch.nn.Module):
    def __init__(self, n_head, input_dim):
        super().__init__()
        self.multi_head_attention = torch.nn.MultiheadAttention(n_head, input_dim)
        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = torch.nn.ReLU()
        self.attention_dropout = torch.nn.Dropout(0.5)
        self.layer_norm1 = torch.nn.LayerNorm(input_dim)
        self.fc2 = torch.nn.Linear(input_dim, input_dim)
        self.tanh = torch.nn.Tanh()
        self.final_layer_norm1 = torch.nn.LayerNorm(input_dim)

    def forward(self, x): 
        x = x.view(1, 1, -1)
        v1 = self.multi_head_attention(x, x, x)[0]
        v2 = v1 + 0.5
        v3 = self.fc1(v2)
        v4 = self.relu(v3)
        v5 = self.attention_dropout(v4)
        v6 = torch.add(v2, 1, v5)
        v7 = self.layer_norm1(v6)
        v8 = self.fc2(v7)
        v9 = self.tanh(v8)
        v10 = self.final_layer_norm1(x + v9)
        return v10

# Initializing the model
m = Model(1, 5)

# Inputs to the model
x = torch.randn(5, 5)
