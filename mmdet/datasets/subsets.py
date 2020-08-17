import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SubsetDataset(CustomDataset):

    def __init__(self, *args, **kwargs):
        self.subset = kwargs.pop('subset')
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        anns = super().load_annotations(ann_file)
        sub_ann = []
        for lower, upper in self.subset:
            sub_ann.extend(anns[int(lower * len(anns)):int(upper * len(anns))])
        return sub_ann



@DATASETS.register_module()
class CrossValDataset(SubsetDataset):

    def __init__(self, *args, **kwargs):
        self.num_folds = kwargs.pop('num_folds')  # Total number of folds to use
        self.cur_fold = kwargs.pop('cur_fold')  # Current fold (zero based, max is num_folds - 1)
        self.train = kwargs.pop('train')  # True: returns (num_folds - 1)/num_folds portion of data
        r = 1.0 / self.num_folds
        subset = [[i * r, (i + 1 * r)] for i in range(self.num_folds)]
        val = subset.pop(self.cur_fold)
        if not self.train:
            subset = [val]
        super().__init__(*args, subset=subset, **kwargs)

