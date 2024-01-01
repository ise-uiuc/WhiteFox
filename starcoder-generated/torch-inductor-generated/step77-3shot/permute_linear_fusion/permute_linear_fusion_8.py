
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for v1 in range(5):
            in_features = 2
            out_features = 12
            self.__setattr__('linear_' + str(v1), torch.nn.Linear(in_features, out_features))
            for v2 in range(2):
                in_features = 12
                out_features = 16
                self.__setattr__('linear_' + str(v1) + str(v2), torch.nn.Linear(in_features, out_features))
                self.__setattr__('ReLU' + str(v1) + str(v2), torch.nn.ReLU6())
                in_features = 16
                out_features = 20
                self.__setattr__('linear_' + str(v1) + str(v2 + 1), torch.nn.Linear(in_features, out_features))
                self.__setattr__('ReLU' + str(v1) + str(v2 + 1), torch.nn.ReLU6())

                in_features = 20
                out_features = 24
                self.__setattr__('linear_' + str(v1) + str(v2 + 2), torch.nn.Linear(in_features, out_features))
                self.__setattr__('ReLU' + str(v1) + str(v2 + 2), torch.nn.ReLU6())
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear_0.weight, self.linear_0.bias)
        v2 = self.ReLU00(v2)
        v3 = torch.nn.functional.linear(v1, self.linear_0.weight, self.linear_0.bias)
        v3 = self.ReLU00(v3)
        v1 = torch.mean(v1, -1)
        v4 = x1.permute(0, 2, 1)
        v5 = torch.nn.functional.linear(v4, self.linear_1.weight, self.linear_1.bias)
        v5 = self.ReLU11(v5)
        v6 = torch.nn.functional.linear(v4, self.linear_1.weight, self.linear_1.bias)
        v6 = self.ReLU11(v6)
        return torch.sum(self.linear_2.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 500)
