from torch.utils.data import DataLoader, random_split

def split_dataset_by_percentage(dataset, batch_size=20, percentage=0.5):
    dataset_size = len(dataset)
    print(dataset_size)
    divison_size = int(dataset_size * percentage)
    print("hello")
    dataset1, dataset2 = random_split(dataset, [divison_size, dataset_size - divison_size])
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)
    return dataloader1, dataloader2