"""Fixed facts about Waterbirds and project-wide constants.

Group encoding follows the original Group DRO code: g = 2 * y + a, with
    y = 0 landbird, 1 waterbird
    a = 0 land background, 1 water background
so groups 0 and 3 are "majority" (background agrees with the class) and
groups 1 and 2 are "minority" (background disagrees with the class).
"""

N_CLASSES = 2
N_GROUPS = 4

GROUP_NAMES = ("landbird/land", "landbird/water", "waterbird/land", "waterbird/water")
MAJORITY_GROUPS = (0, 3)
MINORITY_GROUPS = (1, 2)

# Codes used in the 'split' column of the official metadata.csv.
OFFICIAL_SPLIT_CODES = {"train": 0, "val": 1, "test": 2}

# Per-group counts of the official release (Sagawa et al., 2020), ordered by g.
EXPECTED_OFFICIAL_COUNTS = {
    "train": (3498, 184, 56, 1057),
    "val": (467, 466, 133, 133),
    "test": (2255, 2255, 642, 642),
}
EXPECTED_POOL_SIZE = 11788

# torchvision ImageNet normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Evaluation sets whose outputs are written to the run's sealed/ directory.
# Training code never computes metrics on these; only the final evaluation
# script reads them, after model selection has been done on validation data.
SEALED_SETS = frozenset({"test"})
