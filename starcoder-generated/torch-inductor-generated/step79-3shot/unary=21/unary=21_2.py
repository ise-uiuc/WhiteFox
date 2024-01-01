
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features=28 * 28, out_features=512)
        self.layer_2 = torch.nn.Linear(in_features=512, out_features=256)
        self.layer_3 = torch.nn.Linear(in_features=256, out_features=128)
        self.layer_4 = torch.nn.Linear(in_features=128, out_features=64)
        self.layer_5 = torch.nn.Linear(in_features=64, out_features=10)

    def forward(self, state):
        x = nn.Dropout(0.2)(state)
        x = F.tanh(self.layer_1(x))
        x = torch.tanh(self.layer_2(x))
        x = torch.tanh(self.layer_3(x))
        x = F.softmax(self.layer_4(x), dim=-1)
        return x
# Inputs to the model
state = torch.randn(1, 28 * 28)
