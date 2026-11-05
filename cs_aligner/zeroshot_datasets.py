"""Zero-shot classification test sets, ported from clip-refine (data/generic.py, main/experimental_settings.py).

Every dataset is read from <data_root>/<folder>, data_root being the folder given to vlm-tuning's
download_datasets.sh --data. Labels follow the order of the class name files in cs_aligner/classnames/.
"""
import json
import os

import torchvision.datasets
from PIL import Image
from torch.utils.data import Dataset

CLASSNAMES = os.path.join(os.path.dirname(__file__), "classnames")


def classname_path(name):
    return os.path.join(CLASSNAMES, name)


def read_classnames(classname_file):
    with open(classname_file, "r") as f:
        return [line.strip().split("\t")[1].replace("_", " ") for line in f]


def normalize(name):
    return name.strip().lower().replace("_", " ")


class ImageList(Dataset):
    # (image path, label) samples, labels following the order of classname_file used by the zero-shot classifier

    def __init__(self, samples, classname_file, transform):
        self.samples = samples
        self.classes = read_classnames(classname_file)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


class CoOpSplit(ImageList):
    # Test split of a dataset in the CoOp layout written by vlm-tuning's download_datasets.sh:
    # <root>/<image_dir>/<image path> and <root>/<split_file> = {"test": [[image path, label, class name], ...]}.
    # The split's labels follow CoOp's class order, so samples are relabeled by class name against classname_file.
    image_dir: str
    split_file: str
    classname_file: str
    match_folder = False  # match the class by the image's folder instead of the split's class name

    def __init__(self, transform, root):
        classname_file = classname_path(self.classname_file)
        label_of = {normalize(c): i for i, c in enumerate(read_classnames(classname_file))}
        with open(os.path.join(root, self.split_file), "r") as f:
            split = json.load(f)["test"]
        samples = []
        for path, _, name in split:
            key = normalize(path.split("/")[0] if self.match_folder else name)
            if key not in label_of:
                raise KeyError(f"class '{key}' of {self.split_file} is not in {self.classname_file}")
            samples.append((os.path.join(root, self.image_dir, path), label_of[key]))
        super().__init__(samples, classname_file, transform)


class Aircraft(ImageList):
    # <root> is the data/ folder of fgvc-aircraft-2013b; classname_file follows the order of variants.txt
    classname_file = "fgvc_aircraft.txt"

    def __init__(self, transform, root):
        classname_file = classname_path(self.classname_file)
        with open(os.path.join(root, "variants.txt"), "r") as f:
            variants = [line.strip() for line in f]
        assert len(variants) == len(read_classnames(classname_file))
        samples = []
        with open(os.path.join(root, "images_variant_test.txt"), "r") as f:
            for line in f:
                image_id, variant = line.strip().split(" ", 1)
                samples.append((os.path.join(root, "images", f"{image_id}.jpg"), variants.index(variant)))
        super().__init__(samples, classname_file, transform)


class Car(CoOpSplit):
    image_dir = ""  # the split's paths start with cars_test/
    split_file = "split_zhou_StanfordCars.json"
    classname_file = "stanford_cars.txt"


class Caltech101(CoOpSplit):
    image_dir = "101_ObjectCategories"
    split_file = "split_zhou_Caltech101.json"
    classname_file = "caltech101.txt"
    match_folder = True  # classname_file holds the folder names (CoOp renames a few, e.g. Faces -> face)


class DTD(CoOpSplit):
    image_dir = "images"
    split_file = "split_zhou_DescribableTextures.json"
    classname_file = "dtd.txt"


class EuroSAT(CoOpSplit):
    image_dir = "2750"
    split_file = "split_zhou_EuroSAT.json"
    classname_file = "eurosat.txt"


class Food(CoOpSplit):
    image_dir = "images"
    split_file = "split_zhou_Food101.json"
    classname_file = "food101.txt"


class Flower(CoOpSplit):
    image_dir = "jpg"
    split_file = "split_zhou_OxfordFlowers.json"
    classname_file = "oxford_flowers.txt"


class ImageNet(torchvision.datasets.ImageFolder):
    # <root>/val/<wnid>/*.JPEG: the sorted wnids follow the order of imagenet.txt
    classname_file = "imagenet.txt"

    def __init__(self, transform, root):
        super().__init__(root=os.path.join(root, "val"), transform=transform)


class Pet(CoOpSplit):
    image_dir = "images"
    split_file = "split_zhou_OxfordPets.json"
    classname_file = "oxford_pets.txt"


class SUN397(CoOpSplit):
    image_dir = "SUN397"
    split_file = "split_zhou_SUN397.json"
    classname_file = "sun397.txt"


class UCF101(CoOpSplit):
    image_dir = "UCF-101-midframes"
    split_file = "split_zhou_UCF101.json"
    classname_file = "ucf101.txt"


# result name -> (dataset class, folder under data_root), as in clip-refine's main/experimental_settings.py
DATASETS = {
    "zeroshot_classification_aircraft": (Aircraft, "fgvc_aircraft"),
    "zeroshot_classification_car": (Car, "stanford_cars"),
    "zeroshot_classification_caltech101": (Caltech101, "caltech-101"),
    "zeroshot_classification_dtd": (DTD, "dtd"),
    "zeroshot_classification_eurosat": (EuroSAT, "eurosat"),
    "zeroshot_classification_flower": (Flower, "oxford_flowers"),
    "zeroshot_classification_food": (Food, "food-101"),
    "zeroshot_classification_imagenet": (ImageNet, "imagenet/images"),
    "zeroshot_classification_pet": (Pet, "oxford_pets"),
    "zeroshot_classification_sun": (SUN397, "sun397"),
    "zeroshot_classification_ucf": (UCF101, "ucf101"),
}


def load_dataset(name, data_root, transform):
    cls, folder = DATASETS[name]
    return cls(transform=transform, root=os.path.join(data_root, folder)), classname_path(cls.classname_file)
