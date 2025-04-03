from datasets.proteins_loader import load_proteins_dataset
from datasets.enzymes_loader import load_enzymes_dataset
# 未来可以加更多 loader

def load_dataset(name, batch_size=32):
    name = name.lower()
    if name == 'proteins':
        return load_proteins_dataset(batch_size)
    elif name == 'enzymes':
        return load_enzymes_dataset(batch_size)
    else:
        raise ValueError(f"Unknown dataset: {name}")
