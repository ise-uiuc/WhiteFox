
class Model(torch.nn.Module):
    def __init__(self):
        super(Model):
            super().__init__()
            self.linear = torch.nn.Linear(2, 4)
            self.gelu = torch.nn.GELU()
            self.dropout = torch.nn.Dropout(0.1)
            self.transformer_encoder_layer_stack = torch.nn.TransformerEncoderLayer
            self.transformer = torch.nn.Transformer(self.transformer_encoder_layer_stack(d_model=4, nhead=2, dim_feedforward=6))
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.gelu(v2)
        v4 = v3.permute(1, 0, 2)
        v5 = self.dropout(v4)
        v6 = self.transformer(v5)
        v7 = v6.permute(1, 0, 2)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
