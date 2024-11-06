# MSGRL

This repository includes the scripts for MSGRL.

## Training & Evaluation on Stage_1

```
# Run with default config.
# $DATASET are described below.
python main_pretrain_stage1.py --dataset $DATASET
```

the trained motif encoder will be saved in log/**/model

## Training & Evaluation on Stage_2

```
# Run with default config.
# $DATASET are described below.
python main_stage2.py --dataset $DATASET --model_pretrain_load_path $model
```

### `$model`
`$model` specified the path of the stage1 model save path. It should be one of the followings:
- `log/**/model`

### `$DATASET`
`$DATASET` specified the name of the molecule dataset. It should be one of the followings:
- `ogbg-molbace`
- `ogbg-molbbbp`
- `ogbg-molsider`
- `ogbg-molclintox`
- `ogbg-moltox21`
- `ogbg-molesol`
- `ogbg-molfreesolv`
- `ogbg-mollipo`

