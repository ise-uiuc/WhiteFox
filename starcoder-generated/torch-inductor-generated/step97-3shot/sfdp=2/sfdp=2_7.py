
class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=config["hidden_size"], num_heads=config["attention_heads"])
 
    def forward(self, x1, x2):
        v1, v2 = self.attention(x1, x2)
        return v1, v2

# Initializing the model
config = {
    "hidden_size": 8,
    "attention_heads": 8
}
m = Model(config)

# Inputs to the model
x1 = torch.randn(1, 8, 32)
x2 = torch.randn(1, 8, 64)
