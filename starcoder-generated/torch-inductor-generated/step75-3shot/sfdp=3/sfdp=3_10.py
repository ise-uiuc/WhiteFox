
class Model(torch.nn.Module):
    def __init__(self,
                 num_blocks=2,
                 num_heads=4,
                 hidden_size=64,
                 dropout_p=0.75,
                 num_classes=49 + 1):
        super().__init__()
        self.query_projection, self.key_projection, self.value_projection, self.final_projection = self.build(num_blocks, num_heads, hidden_size, dropout_p, num_classes)
 
    def build(self, num_blocks, num_heads, hidden_size, dropout_p, num_classes):
        scale_factor = hidden_size ** -0.5
        all_blocks = []
        for _ in range(num_blocks):
            all_blocks.append(torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                              nhead=num_heads,
                                                              dim_feedforward=hidden_size,
                                                              dropout=dropout_p))
        block = torch.nn.Sequential(*all_blocks)
        return block
 
    def forward(self, x1, x2):
        q = self.query_projection(x1)
        k = self.key_projection(x2)
        v = self.value_projection(x2)
        logits = self.final_projection(self.block(x1, x2))
        return logits

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 448)
x2 = torch.randn(1, 256, 448)
