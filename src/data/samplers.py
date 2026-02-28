from torch.utils.data import Sampler
import random
import torch

class WeightedSampler(Sampler):
    def __init__(self, data_source, weights_config=None):
        """
        weights_config: List trọng số từ file config (VD: [1.0, 3.0, 1.0, 3.0])
        """
        self.num_samples = len(data_source.image_labels)
        
        # 1. Lấy danh sách toàn bộ label trong dataset
        all_labels = [label for _, label in data_source.image_labels]
        all_labels = torch.tensor(all_labels)

        # 2. Xác định trọng số cho từng mẫu
        if weights_config is not None:
            # Sử dụng trọng số tùy chỉnh từ config của bạn
            class_weights = torch.tensor(weights_config).float()
        else:
            # Tự động tính trọng số nghịch đảo nếu không có config
            counts = torch.bincount(all_labels)
            class_weights = 1.0 / counts.float()

        # 3. Tạo list trọng số cho từng ảnh một
        self.sample_weights = class_weights[all_labels]

    def __iter__(self):
        # Multinomial sẽ chọn index dựa trên xác suất (trọng số)
        # replacement=True cho phép một ảnh lớp hiếm được xuất hiện nhiều lần trong 1 epoch
        indices = torch.multinomial(self.sample_weights, self.num_samples, replacement=True)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:
        if j >= num_samples_cls:
            j = 0
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        i += 1
        j += 1


class RandomCycleIter:
    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


class ClassAwareSampler(Sampler):
    def __init__(
        self,
        data_source,
        num_samples_cls=4,
    ):
        # pdb.set_trace()
        num_classes = data_source.num_classes
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, (_, label) in enumerate(data_source.image_labels):
            cls_data_list[label].append(i)

        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        # self.num_samples = sum([len(x) for x in cls_data_list])
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(
            self.class_iter, self.data_iter_list, self.num_samples, self.num_samples_cls
        )

    def __len__(self):
        return self.num_samples


class DownSampler(Sampler):
    def __init__(self, data_source, n_max=100):
        self.num_classes = data_source.num_classes
        self.cls_data_list = [list() for _ in range(self.num_classes)]
        for i, image_label in enumerate(data_source.image_labels):
            _, label = image_label
            self.cls_data_list[label].append(i)

        self.n_max = n_max
        self.cls_num_list = [min(n_max, len(x)) for x in self.cls_data_list]
        self.num_samples = sum(self.cls_num_list)

    def __iter__(self):
        data_list = []
        for y in range(self.num_classes):
            random.shuffle(self.cls_data_list[y])
            data_list.extend(self.cls_data_list[y][: self.n_max])
        random.shuffle(data_list)

        for i in range(self.num_samples):
            yield data_list[i]

    def __len__(self):
        return self.num_samples
