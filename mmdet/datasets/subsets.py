import mmcv
import numpy as np
from multiprocessing import Pool
import itertools
import sklearn.metrics as skm

from .builder import DATASETS
from .custom import CustomDataset
from ..core.evaluation.mean_ap import get_cls_results
from ..core.evaluation.bbox_overlaps import bbox_overlaps


@DATASETS.register_module()
class SubsetDataset(CustomDataset):

    def __init__(self, *args, **kwargs):
        self.subset = kwargs.pop('subset')
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        anns = super().load_annotations(ann_file)
        sub_ann = []
        if isinstance(self.subset[0],(int,float)):
            self.subset = [self.subset]
        for lower, upper in self.subset:
            sub_ann.extend(anns[int(lower * len(anns)):int(upper * len(anns))])
        return sub_ann
    def evaluate(self,
                 results,
                 metric='imageList',
                 iou_thr=0.05,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
        """

        if isinstance(metric, str):
            metric = [metric]
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if "imageList" in metric:
            metric.remove("imageList")
            eval_two_by_two_table(
                results,
                annotations,
                iou_thr=iou_thr
            )
        for m in metric:
            er = super().evaluate(results, metric=m, iou_thr=iou_thr, **kwargs)
            eval_results.update(er)
        return eval_results



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


def eval_two_by_two_table(det_results,
                          annotations,
                          iou_thr=0.1):

    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = 1   # len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    pool = Pool()
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        twobytwo = itertools.starmap(
            twobytwo_func,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)])
        )
        cls_res = enumerate([*twobytwo])
        cls_res = np.vstack(list(map(rowcombine,cls_res)))
        adj_res = threshres(cls_res, 0.5) #TODO: select threshold based on ROC curve
        clinstats(cls_res, 0.5)
        with np.printoptions(precision=3):
            errors = adj_res[np.where(np.logical_or(adj_res[:,2]>0, adj_res[:,4]>0))]
            print(errors)
            print('* '.join(f'{int(i):03d}' for i in errors[:, 0]))


        with np.printoptions(threshold=np.inf):
            pass
            #print(cls_res)
        # tp, fp, tn, fn,score = tuple(zip(*twobytwo))
        # print(np.hstack(r) for r in [tp, fp, tn, fn])

def rowcombine(r):
    vals = np.vstack(r[1]).T
    inds = np.vstack((r[0] for i in range(vals.shape[0])))
    return np.hstack((inds,vals))

def plotROC(gt,scores): # this is scratch space for plot code
    #import scikitplot as skp
    #import matplotlib.pyplot as plt
    gt = cls_res[:, 1] + cls_res[:, 4]
    probs = np.vstack([1 - cls_res[:, 5], cls_res[:, 5]]).T
    skp.metrics.plot_roc(gt, probs, plot_micro=False, plot_macro=False, classes_to_plot=[1.0])
    plt.show()

def clinstats(m, t):
    m_adj = threshres(m, t)
    ind, tp, fp, tn, fn, score = m_adj.sum(axis=0)
    print(f"Sensitivity: {tp/(tp+fn)}")
    print(f'Specificity: {tn/(fp+tn)}')
    print(f'*PPV: {tp/(tp+fp)}  *NPV: {tn/(fn+tn)}')
    gt = m[:, 1] + m[:, 4]
    print(f'ROC AUC: {skm.roc_auc_score(gt,m[:,5])}')

def threshres(m, t):
    def adjrowthresh(row):
        ind, tp, fp, tn, fn, score = row
        if score < t:
            if fp > 0:
                fp = 0.0
                tn = 1.0
            elif tp > 0:
                tp = 0.0
                fn = 1.0
        return np.array((ind, tp, fp, tn, fn, score))
    return np.apply_along_axis(adjrowthresh, axis=1, arr=m)

def twobytwo_func(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.1,
                 area_ranges=None):
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, np.maximum(1, num_dets)), dtype=np.float32)
    fp = np.zeros((num_scales, np.maximum(1, num_dets)), dtype=np.float32)
    tn = np.zeros((num_scales, np.maximum(1, num_dets)), dtype=np.float32)
    fn = np.zeros((num_scales, np.maximum(1, num_dets)), dtype=np.float32)
    score = np.zeros((num_scales, np.maximum(1, num_dets)), dtype=np.float32)

    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            if det_bboxes.shape[0] == 0:
                tn[...] = 1
            else:
                score[...] = det_bboxes[:,4]
                fp[...] = 1
        else:
            assert False, "Area ranges not implemented"
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                det_bboxes[:, 3] - det_bboxes[:, 1])
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp, tn, fn, score

    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else: # area ranges
            assert False
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds[:1]:   #Hack to ignore all but highest prob det
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    score[k, i] = det_bboxes[matched_gt, 4]
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else: #area ranges
                assert False
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
        # Count a fn for every uncovered gt (doesn't account for area ranges)
        # fn[k,0] = gt_covered.size - np.count_nonzero(gt_covered)
        if np.count_nonzero(gt_covered) == 0:
            fn[k, 0] = 1
    return tp, fp, tn, fn, score
