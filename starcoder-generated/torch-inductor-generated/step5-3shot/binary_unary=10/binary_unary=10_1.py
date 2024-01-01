

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(224, 256)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + torch.ones(v1.size())
        v3 = torch.nn.functional.relu(v2)
        return v3
class MyDataset(Dataset):

    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        return sample, target

from torch.utils.data import DataLoader

from torchsummary import summary
training_data_path = 'path_to_training_samples.pt'
training_labels_path = 'path_to_training_labels.pt'
testing_data_path = 'path_to_training_samples.pt'
testing_labels_path = 'path_to_training_labels.pt'

train_data = torch.load(training_data_path)
train_target = torch.load(training_labels_path)
test_data = torch.load(testing_data_path)
test_target = torch.load(testing_labels_path)

train_dataset = MyDataset(train_data, train_target)
test_dataset = MyDataset(test_data, test_target)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
image, label = next(iter(test_loader))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(224, 256)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + torch.ones(v1.size())
        v3 = torch.nn.functional.relu(v2)
        return v3
 
n = Model()
print(summary(n, input_size=(3, 224, 224), batch_size=1, device="cpu"))
n.train()

