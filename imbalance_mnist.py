import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import gzip

class MNIST(torchvision.datasets.MNIST):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        training_images_unzipped = gzip.open('train-images-idx3-ubyte.gz', 'r')
        training_labels_unzipped = gzip.open('train-labels-idx1-ubyte.gz', 'r')
        test_images_unzipped = gzip.open('t10k-images-idx3-ubyte.gz', 'r')
        test_labels_unzipped = gzip.open('t10k-labels-idx1-ubyte.gz', 'r')

        image_size = 28
        training_labels_unzipped.read(8)
        label_buf = training_labels_unzipped.read(60000)
        labelArray = np.frombuffer(label_buf, dtype=np.uint8).astype(np.int32)

        training_images_unzipped.read(16)
        buf = training_images_unzipped.read(image_size * image_size * 60000)
        imbalanced_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)


        test_labels_unzipped.read(8)
        test_label_buf = test_labels_unzipped.read(10000)
        test_labelArray = np.frombuffer(test_label_buf, dtype=np.uint8).astype(np.int32)

        test_images_unzipped.read(16)
        test_images_buf = test_images_unzipped.read(image_size * image_size * 10000)
        test_images = np.frombuffer(test_images_buf, dtype=np.uint8).astype(np.float32)

        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class CIFAR100(MNIST):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()