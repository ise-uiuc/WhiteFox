
class MyModule4_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleDict({
        '0': torch.nn.Conv2d(3, 5, 2, stride=(2, 1), padding=(2, 1)),
        '1': torch.nn.Conv2d(5, 5, 2, stride=(2, 2), padding=(1, 1), bias=False),
        })
        self.features.update({'2': torch.nn.Conv2d(5, 5, 2, stride=(2, 1), padding=(4, 1))})
        self.features.update({'3': torch.nn.Conv2d(5, 7, 2, stride=(2, 2), padding=(3, 1))})
        self.regressor = torch.nn.Linear(5, 15)
    def forward(self, x):
        negative_slope = 1.8414381
        v2 = self.features['0'](x)
        v6 = v2 > 0
        v7 = v2 * negative_slope
        v9 = torch.where(v6, v2, v7)
        v10 = self.features['1'](v9)
        v3 = v10 > 0
        v4 = v10 * negative_slope
        v105 = torch.where(v3, v10, v4)
        v12 = self.features['2'](v105)
        v5 = v12 > 0
        v6 = v12 * negative_slope
        v96 = torch.where(v5, v12, v6)
        v14 = self.features['3'](v96)
        v50 = v14 > 0
        v51 = v14 * negative_slope
        v53 = v14 > 0
        v54 = v14 * negative_slope
        v43 = torch.where(v50, v14, v51)
        v44 = torch.where(v53, v14, v54)
        return (v43 + v44, self.regressor(v2))
    def trainable_named_children(self):
        return [("features", self.features), ("regressor", self.regressor)]
# Inputs to the model
x1 = torch.randn(2, 3, 16, 16)
