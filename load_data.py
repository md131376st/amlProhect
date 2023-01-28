import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import json

CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}

DOMAINS = {
    'art_painting': 0,
    'cartoon': 1,
    'sketch': 1,
    'photo': 1
}


class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y


class PACSDatasetDomainDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y, z = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, z


class PACSDatasetClipDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y, z, t = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, z, t


def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
    return examples


def read_lines_domain_disentangle(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        domain_idx = DOMAINS[domain_name]
        domain_category = (domain_idx, category_idx)
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if domain_category not in examples.keys():
            examples[domain_category] = [image_path]
        else:
            examples[domain_category].append(image_path)
    return examples


def read_lines_clip_disentangle(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        domain_idx = DOMAINS[domain_name]
        domain_category = (domain_idx, category_idx)
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if domain_category not in examples.keys():
            examples[domain_category] = [image_path]
        else:
            examples[domain_category].append(image_path)
    return examples


def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in
                              source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in
                              source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]

    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def build_splits_domain_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines_domain_disentangle(opt['data_path'], source_domain)
    target_examples = read_lines_domain_disentangle(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = dict()
    for domain_category, examples_list in source_examples.items():
        if domain_category[1] not in source_category_ratios.keys():
            source_category_ratios[domain_category[1]] = len(examples_list)
        else:
            source_category_ratios[domain_category[1]] += len(examples_list)
    # source_category_ratios = {domain_category[1]: len(examples_list) for domain_category, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in
                              source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for domain_category, examples_list in source_examples.items():
        domain_idx = domain_category[0]
        category_idx = domain_category[1]
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append(
                    [example, category_idx, domain_idx])  # each pair is [path_to_img, class_label, domain_label]
            else:
                val_examples.append(
                    [example, category_idx, domain_idx])  # each pair is [path_to_img, class_label, domain_label]

    for domain_category, examples_list in target_examples.items():
        domain_idx = domain_category[0]
        category_idx = domain_category[1]
        for example in examples_list:
            test_examples.append(
                [example, category_idx, domain_idx])  # each pair is [path_to_img, class_label, domain_label]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetDomainDisentangle(train_examples, train_transform),
                              batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDomainDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDomainDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=True)

    return train_loader, val_loader, test_loader


def read_label_file(path, source, target):
    label_file = open(f'{path}/labelFile.json')
    data = json.load(label_file)
    source_list = []
    target_list = []
    for info in data:
        if str(info["image_name"]).startswith(source):
            source_list.append(info)
        elif str(info["image_name"]).startswith(target):
            target_list.append(info)
        else:
            continue
    return source_list, target_list


def get_label_info(info_list, target_address):
    value = " "
    target_address = target_address[len("data/PACS/kfold/"):]
    info = list(
        item for item in info_list if item["image_name"] == target_address)
    for item in info:
        #hi = ' '.join(item["descriptions"])
        return  ' '.join(item["descriptions"])
    return value


def build_splits_clip_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines_domain_disentangle(opt['data_path'], source_domain)
    target_examples = read_lines_domain_disentangle(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = dict()
    for domain_category, examples_list in source_examples.items():
        if domain_category[1] not in source_category_ratios.keys():
            source_category_ratios[domain_category[1]] = len(examples_list)
        else:
            source_category_ratios[domain_category[1]] += len(examples_list)
    # source_category_ratios = {domain_category[1]: len(examples_list) for domain_category, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in
                              source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation
    # read label file
    source_label, target_label = read_label_file(opt['data_path'], source_domain, target_domain)
    train_examples = []
    val_examples = []
    test_examples = []

    for domain_category, examples_list in source_examples.items():
        domain_idx = domain_category[0]
        category_idx = domain_category[1]
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            label = get_label_info(source_label, example)
            #label_token= clip.tokenize(label)
            if i > split_idx:

                train_examples.append(
                    [example, category_idx, domain_idx, label])  # each pair is [path_to_img, class_label, target_label]
            else:
                val_examples.append(
                    [example, category_idx, domain_idx, label])  # each pair is [path_to_img, class_label, target_label]

    for domain_category, examples_list in target_examples.items():
        domain_idx = domain_category[0]
        category_idx = domain_category[1]
        for example in examples_list:
            label = get_label_info(target_label, example)
            #label_token = clip.tokenize(label)
            test_examples.append(
                [example, category_idx, domain_idx, label])  # each pair is [path_to_img, class_label, target_label]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetClipDisentangle(train_examples, train_transform),
                              batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetClipDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetClipDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=True)

    return train_loader, val_loader, test_loader
