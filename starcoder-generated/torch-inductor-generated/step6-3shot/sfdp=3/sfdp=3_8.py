
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=[512, 14, 14])
        self.dropout_1 = torch.nn.Dropout(p=0.1)
        self.linear_1 = torch.nn.Linear(in_features=2048, out_features=512)
        self.dropout_2 = torch.nn.Dropout(p=0.1)
        self.linear_2 = torch.nn.Linear(in_features=512, out_features=512)

    def forward(self, x1):
        h1 = self.layer_norm(x1)
        h2 = self.dropout_1(h1)
        h3 = self.linear_1(h2)
        h4 = self.dropout_2(h3)
        h5 = self.linear_2(h4)
        return h5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2048, 512)
