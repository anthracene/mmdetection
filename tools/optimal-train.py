import subprocess
import re

GPU_COUNT = 4
CONFIG = 'configs/IVCFilter/IVCcascade50.py'

params = {'bc_prob': 0.27780673582577337, 'brightness_ctr': -0.1085989224361768, 'brightness_rng': 0.09168599715913388,
          'contrast_ctr': 0.03891045164390627, 'contrast_rng': 0.016499312414779345, 'flip_prob': 0.03221964565065233,
          'learning_rate': 0.008007698458933624, 'rot_fine_limit': 17, 'rotate_90_prob': 0.12526766891192068,
          'rotate_fine_prob': 0.3117198734813435}

locals().update(params)

brightness_min = brightness_ctr - (brightness_rng / 2)
brightness_max = brightness_ctr + (brightness_rng / 2)
contrast_min = contrast_ctr - (contrast_rng / 2)
contrast_max = contrast_ctr + (contrast_rng / 2)

metric_results = []
with open("/tmp/aucs", "a") as aucfile:

    for trial in range(10):
        subprocess.run(['tools/dist_train.sh',
                        CONFIG, str(GPU_COUNT),
                        '--work-dir', f'work_dirs/full-train/{trial:03d}',
                        '--options', f'optimizer.lr={learning_rate}',
                        f'data.train.pipeline[4].transforms[1].p={rotate_90_prob}',
                        f'data.train.pipeline[4].transforms[0].p={bc_prob}',
                        f'data.train.pipeline[4].transforms[0].contrast_limit={contrast_min},{contrast_max}',
                        f'data.train.pipeline[4].transforms[0].brightness_limit={brightness_min},{brightness_max}',
                        f'data.train.pipeline[3].flip_ratio={flip_prob}',
                        f'data.train.pipeline[4].transforms[2].p={rotate_fine_prob}',
                        f'data.train.pipeline[4].transforms[2].limit={rot_fine_limit}'])

        test_out = subprocess.run(['tools/dist_test.sh',
                                   CONFIG,
                                   f'work_dirs/full-train/{trial:03d}/latest.pth', str(GPU_COUNT),
                                   '--eval', 'imageList'],
                                  stdout=subprocess.PIPE).stdout.decode('utf-8')
        m = re.search('^ROC AUC: (\d+.\d+)', test_out, re.MULTILINE)
        result = float(m.group(1))
        metric_results.append(result)
        aucfile.write(f"{result}\n")
        aucfile.flush()
print("AUCs:")
print(metric_results)
