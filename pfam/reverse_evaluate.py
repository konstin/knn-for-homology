"""Does T5 differentiate between real and random domains? Yes, unfortunately"""

# %%

from random import Random

import matplotlib.pyplot as plt
import numpy
from pandas import DataFrame
from sklearn.decomposition import PCA

from pfam.pfam_shared import project_root
from pfam.reverse_shared import forward, reverse, shuffle
from seqvec_search.utils import endfig

# %%

figures = project_root.joinpath("more_sensitive/scrambled-figures")
lines = forward.read_text().splitlines()
ids = [line.split(" ")[1] for line in lines[::2]]

random = Random(42)

# %%

scrambled_name = "reversed"

forward_data = numpy.load(forward.with_suffix(".npy"))
reverse_data = numpy.load(reverse.with_suffix(".npy"))
shuffle_data = numpy.load(shuffle.with_suffix(".npy"))

# %%

both_pca_reverse = PCA(n_components=2).fit_transform(
    numpy.concatenate((forward_data, reverse_data), axis=0)
)
original_pca_reverse = both_pca_reverse[: len(forward_data)]
scrambled_pca_reverse = both_pca_reverse[len(forward_data) :]

both_pca_shuffle = PCA(n_components=2).fit_transform(
    numpy.concatenate((forward_data, shuffle_data), axis=0)
)
original_pca_shuffle = both_pca_shuffle[: len(forward_data)]
scrambled_pca_shuffle = both_pca_shuffle[len(forward_data) :]

# %%

threshold = 0

for name, original, scrambled in [
    ("reversed", original_pca_reverse, scrambled_pca_reverse),
    ("shuffled", original_pca_shuffle, scrambled_pca_shuffle),
]:
    print(
        f"|          | PC1<{threshold}   | PC1>{threshold} \n"
        f"|      --- | ---    | ---\n"
        f"| original | {(original[:, 0] < threshold).sum():<6} | {(original[:, 0] > threshold).sum():<6}\n"
        f"| {name:<8} | {(scrambled[:, 0] < threshold).sum():<6} | {(scrambled[:, 0] > threshold).sum():<6}"
    )

# %%

f, (ax1, ax2) = plt.subplots(1, 2)

DataFrame(
    {"original": original_pca_reverse[:, 0], "reversed": scrambled_pca_reverse[:, 0]}
).plot.density(ax=ax1)
DataFrame(
    {"original": original_pca_shuffle[:, 0], "shuffled": scrambled_pca_shuffle[:, 0]}
).plot.density(ax=ax2)
ax1.set_xlabel("Principal Component 1")
ax2.set_xlabel("Principal Component 1")
ax1.text(0.05, 0.9, "A", transform=ax1.transAxes, size=20, weight="bold")
ax2.text(0.05, 0.9, "B", transform=ax2.transAxes, size=20, weight="bold")
plt.tight_layout()
endfig(figures, "pca_density")

# %%

plt.scatter(
    original_pca_reverse[:, 0],
    original_pca_reverse[:, 1],
    s=0.3,
    alpha=0.3,
    label="original",
)
plt.scatter(
    scrambled_pca_reverse[:, 0],
    scrambled_pca_reverse[:, 1],
    s=0.3,
    alpha=0.3,
    label=scrambled_name,
)
plt.legend()
plt.tight_layout()
endfig(figures, f"pca_{scrambled_name}_both")

# %%

pca = PCA(n_components=2)
pca.fit(numpy.concatenate((forward_data, (reverse_data)), axis=0))

# %%

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(numpy.argsort(numpy.abs(pca.components_[0])))
best_dimension = numpy.argsort(numpy.abs(pca.components_[0]))[-1]

# %%

DataFrame(
    {
        "original": forward_data[:, best_dimension],
        scrambled_name: reverse_data[:, best_dimension],
    }
).plot.density()
plt.show()

# %%

for dim in numpy.argsort(numpy.abs(pca.components_[0]))[::-1][:3]:
    t = (forward_data[:, dim].mean() + reverse_data[:, dim].mean()) / 2
    print(dim, t)
    forward_true = (forward_data[:, dim] < t).sum()
    forward_false = (forward_data[:, dim] > t).sum()
    reverse_false = (reverse_data[:, dim] < t).sum()
    reverse_true = (reverse_data[:, dim] > t).sum()
    print(
        f"{forward_true} {forward_false} {forward_true / (forward_true + forward_false):.0%}"
    )
    print(
        f"{reverse_false} {reverse_true} {reverse_true / (reverse_true + reverse_false):.0%}"
    )

# %%

for dim in numpy.argsort(numpy.abs(pca.components_[0]))[::-1][:3]:
    DataFrame(
        {"original": forward_data[:, dim], scrambled_name: reverse_data[:, dim]}
    ).plot.density()
    plt.show()
