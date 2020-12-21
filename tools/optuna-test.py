import optuna
import subprocess
import re

GPU_COUNT = 4
CONFIG = 'configs/IVCFilter/IVCcascade50.py'
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    rotate_90_prob = trial.suggest_uniform('rotate_90_prob', 0.05, 0.20)
    bc_prob = trial.suggest_uniform('bc_prob', 0.00, 0.50)
    brightness_ctr = trial.suggest_uniform('brightness_ctr', -0.15, 0.15)
    brightness_rng = trial.suggest_uniform('brightness_rng', 0.0, 0.3)
    contrast_ctr = trial.suggest_uniform('contrast_ctr', -0.15, 0.15)
    contrast_rng = trial.suggest_uniform('contrast_rng', 0.0, 0.5)
    flip_prob = trial.suggest_uniform('flip_prob', 0.0, 0.25)
    rot_fine_prob = trial.suggest_uniform('rotate_fine_prob', 0.00,0.50)
    rot_fine_limit = trial.suggest_int('rot_fine_limit', 0, 30)


    brightness_min = brightness_ctr - (brightness_rng/2)
    brightness_max = brightness_ctr + (brightness_rng/2)
    contrast_min = contrast_ctr - (contrast_rng/2)
    contrast_max = contrast_ctr + (contrast_rng/2)

    print(f'Trial number {trial.number}')
    subprocess.run(['tools/dist_train.sh',
                    CONFIG, str(GPU_COUNT),
                    '--work-dir', f'work_dirs/hyperparam/{trial.number:03d}',
                    '--options',  f'optimizer.lr={learning_rate}',
                    f'data.train.pipeline[4].transforms[1].p={rotate_90_prob}',
                    f'data.train.pipeline[4].transforms[0].p={bc_prob}',
                    f'data.train.pipeline[4].transforms[0].contrast_limit={contrast_min},{contrast_max}',
                    f'data.train.pipeline[4].transforms[0].brightness_limit={brightness_min},{brightness_max}',
                    f'data.train.pipeline[3].flip_ratio={flip_prob}',
                    f'data.train.pipeline[4].transforms[2].p={rot_fine_prob}',
                    f'data.train.pipeline[4].transforms[2].limit={rot_fine_limit}'])

    test_out = subprocess.run(['tools/dist_test.sh',
                               CONFIG,
                               f'work_dirs/hyperparam/{trial.number:03d}/latest.pth', str(GPU_COUNT),
                               '--eval', 'imageList'],
                              stdout=subprocess.PIPE).stdout.decode('utf-8')
    m = re.search('^ROC AUC: (\d+.\d+)', test_out, re.MULTILINE)
    result = float(m.group(1))
    return result

if __name__ == '__main__':
    study = optuna.create_study(study_name='IVCF', direction='maximize',
                                storage='sqlite:///work_dirs/optuna.db', load_if_exists=True)
    study.optimize(objective, n_trials=100)
    print(study.best_trial)

