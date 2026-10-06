# HW2 — ZerO Initialization (arXiv:2110.12661)

Воспроизведение экспериментов статьи *ZerO Initialization: Initializing Neural Networks with only Zeros and Ones*: ResNet на CIFAR-10 / ImageNet, сверхглубокие ResNet без BN, Transformer на WikiText-2, анализ рангов, прунинг и Tucker-2.

## Структура

```
hw2/src/
  zero_init.py    # Algorithm 1 (матрицы), Algorithm 2 (свёртки), ZerO для слоёв Transformer
  inits.py        # zero | kaiming | xavier | rezero (CNN); + standard (Transformer)
  resnet.py       # ResNet + опциональный ReZero-gate α на residual-ветке
  transformer.py  # LM: стандартный / ZerO / ReZero (RZTX) encoder
  data.py         # CIFAR-10, ImageNet (ImageFolder), WikiText-2
  metrics.py      # top-1 error, perplexity, stable rank, kernel rank, mean ± std
  training.py     # обучение CNN, траектории stable rank, kernel rank
  lm_training.py  # обучение LM, перплексия
  compression.py  # magnitude pruning, Tucker-2
  train.py        # CLI: CNN
  train_lm.py     # CLI: Transformer
  compress.py     # CLI: прунинг + Tucker-2 по чекпоинтам
  report.py       # таблицы (mean ± std) и графики
```

Зависимости в корневом `pyproject.toml` (`uv sync`). Все команды запускаются из корня репозитория.

## Эксперименты

### Основной скрипт: k запусков × {ZerO, Kaiming, Xavier, ReZero}

```bash
.venv/bin/python hw2/src/run_experiments.py --dataset cifar10 -k 10 --amp
# только ReZero (сравнение со статьёй ReZero / Table 3 ZerO):
.venv/bin/python hw2/src/run_experiments.py --dataset cifar10 --inits rezero -k 5 --amp
```

**ReZero** ([arXiv:2003.04887](https://arxiv.org/abs/2003.04887), [repo](https://github.com/majumderb/rezero)): в каждом residual-блоке `x ← x + α F(x)` с `α = 0` на старте; веса свёрток — Kaiming, BN сохраняется. Для Transformer — слой без LayerNorm и с общим `resweight` на attention+FFN (как RZTX).

Гиперпараметры берутся из статьи для выбранного датасета: CIFAR-10 — ResNet-18, ImageNet — ResNet-50. Опции: `--inits` (подмножество), `--depth`, `--epochs` (для быстрой проверки), `--exp-dir`.

Результат в `hw2/experiments/<dataset>_resnet<depth>/`:

| Файл | Содержимое |
|---|---|
| `models/last_<tag>_seed<N>.pt` | обученные модели |
| `runs/run_<tag>_seed<N>.json` | лог каждого запуска |
| `metrics.json` | все метрики: гиперпараметры, история по эпохам, финальные и лучшие значения, kernel/stable rank, mean ± std по seed |
| `summary.md` | таблица: top-1 test error и accuracy, mean ± std |
| `training_curves.png` | loss, top-1 error, test accuracy по эпохам (mean ± std), финальная accuracy |
| `quality_metrics.png` | top-1 test error и accuracy (финальная и лучшая эпоха), mean ± std, точки — отдельные seed |
| `stable_rank.png` | stable rank `layer{2,3,4}.0.conv1` по итерациям (Fig. 5): для `W − W_ZerO` и для `W` |
| `kernel_ranks.png` | ранги ядер всех свёрточных слоёв |

Перестроить графики из готового `metrics.json`, не обучая заново: `--plot-only --exp-dir <папка>`.

### Table 2–3: ResNet-18 / ResNet-50 на CIFAR-10, 10 сидов

```bash
.venv/bin/python hw2/src/train.py --depth 18 --inits zero kaiming xavier --seeds 0 1 2 3 4 5 6 7 8 9 --amp
.venv/bin/python hw2/src/train.py --depth 50 --inits zero kaiming xavier --seeds 0 1 2 3 4 5 6 7 8 9 --amp
```

Гиперпараметры по статье: SGD lr=0.1, momentum=0.9, weight decay=1e-4, линейный warmup 10 эпох (по итерациям). Расписание после warmup в статье не указано — используется стандартное: 200 эпох, ×0.1 на 100 и 150.

Для каждого запуска сохраняются top-1 test error и accuracy (финальная и лучшая эпоха); `report.py` сводит их в `mean ± std` по сидам.

ImageNet (ResNet-50, warmup 5, 90 эпох): `--dataset imagenet --depth 50 --data-dir <path>` с `train/` и `val/` в формате ImageFolder.

### Глубина ResNet и Fig. 4: сверхглубокие сети без BN

`--depth` принимает 18/34/50/101/152 (конфигурации He et al.) или любую глубину `8n+2` (basic) / `12n+2` (`--block bottleneck`). `--norm none` заменяет BN обучаемыми скалярами scale/bias (как в Fixup).

```bash
.venv/bin/python hw2/src/train.py --depth 498 --norm none --inits zero kaiming --epochs 15 --amp
```

### Table 4: Transformer на WikiText-2

```bash
.venv/bin/python hw2/src/train_lm.py --layers 2 4 6 8 10 20 --inits standard zero
```

Модель и обучение как в PyTorch `word_language_model`: d_model=200, 2 головы, SGD lr=20, клиппинг 0.25, 20 эпох с одним понижением lr на 10-й эпохе. ZerO: `W_Q = I`, `W_K = W_V = 0`, feed-forward и out-проекция по Algorithm 1. Метрика — test perplexity модели с лучшей val perplexity.

### Fig. 5–6: ранги, прунинг, Tucker-2

- **Stable rank** первой свёртки 2-й, 3-й и 4-й групп residual-блоков (`layer{2,3,4}.0.conv1`) логируется каждые `--rank-log-interval` итераций в `rank_trajectory`.
- **Kernel rank** всех свёрток после обучения сохраняется в `kernel_ranks`.
- Для обоих сохраняются ранг весов `W` и ранг residual-компоненты `W − W_ZerO`, как в коде авторов (они считают ранг `W − I`). У ZerO stable rank самих весов на старте максимальный (identity / Hadamard), поэтому траектория «от низкого ранга к высокому» видна только на residual-компоненте.

```bash
.venv/bin/python hw2/src/compress.py hw2/results/last_cifar10_resnet18_bn_zero_seed0.pt \
                                     hw2/results/last_cifar10_resnet18_bn_kaiming_seed0.pt
```

Magnitude pruning: в каждом Conv2d/Linear обнуляется доля весов с наименьшим |w|, без дообучения. Tucker-2: разложение по канальным модам свёртки `layer4.1.conv1` (512→512), HOSVD + HOOI, с перебором рангов.

### Отчёт

```bash
.venv/bin/python hw2/src/report.py
```

Пишет `hw2/results/tables.md` и графики в `hw2/results/figures/`.

## Отличия от текста статьи

- Нормировка Hadamard `2^{-m/2}` (ортонормированная, как в коде авторов); в тексте статьи напечатано `2^{-(m-1)/2}`.
- Embedding и выходной слой Transformer инициализируются как в примере PyTorch: статья применяет ZerO только к attention и feed-forward внутри слоёв.
