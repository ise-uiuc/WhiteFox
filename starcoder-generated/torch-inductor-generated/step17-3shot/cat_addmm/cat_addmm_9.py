
class Model(nn.Module):
    def __init__(self, num_features_out=6, p=0.5, training=True):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=p, inplace=False)
        self.linear = nn.Linear(2, 4)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = F.log_softmax(x, dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
