from torch_geometric.datasets import TUDataset

def load_enzymes_dataset(batch_size=32, shuffle=True):
    dataset = TUDataset(root='data/ENZYMES', name='ENZYMES')
    dataset = dataset.shuffle()
    total = len(dataset)
    train_len = int(0.8 * total)
    val_len = int(0.1 * total)

    train_dataset = dataset[:train_len]
    val_dataset = dataset[train_len:train_len+val_len]
    test_dataset = dataset[train_len+val_len:]

    from torch_geometric.loader import DataLoader
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )
