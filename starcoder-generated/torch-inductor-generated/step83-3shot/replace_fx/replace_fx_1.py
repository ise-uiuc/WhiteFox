
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.relu(x)
        x = F.dropout(x, p=0.05, training=True)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.05, training=True)
        x = torch.nn.functional.relu(x)
        x = F.dropout(x, p=0.05, training=True)
        x = F.dropout(x, p=0.05, training=True)
        x = torch.nn.functional.relu(x)
        x = F.dropout(x, p=0.05, training=True)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
