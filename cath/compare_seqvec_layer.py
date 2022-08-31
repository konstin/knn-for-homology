# %%

import itertools
import json
import sys
from concurrent.futures.process import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy
import ternary
from numpy import ndarray
from tqdm import tqdm

from cath.cath_shared import cath_data, load_mapping, figures
from cath.search import search
from seqvec_search.utils import endfig

if len(sys.argv) == 2:
    granularity = int(sys.argv[1])
else:
    granularity = 5

# %%

cnn = numpy.load(cath_data.joinpath("SeqVec CharCNN.npy"))
lstm1 = numpy.load(cath_data.joinpath("SeqVec LSTM1.npy"))
lstm2 = numpy.load(cath_data.joinpath("SeqVec LSTM2.npy"))

# %%

ids = numpy.asarray(
    [
        i.split("|")[2].split("/")[0]
        for i in json.loads(cath_data.joinpath("ids.json").read_text())
    ]
)

mapping_levels, mapping_array = load_mapping(ids)

# %%

# Get all possible linear combinations in our grid
# by generating all combinations and filtering to those that add up to 1
ratios = numpy.asarray(
    list(itertools.product(numpy.linspace(0, 1, granularity), repeat=3))
)
ratios = ratios[numpy.isclose(ratios.sum(axis=1), 1)]

# %%

hits_per_method = dict()
scores_per_method = dict()

linear_combinations = []
for ratio in ratios:
    linear_combinations.append(ratio[0] * cnn + ratio[1] * lstm1 + ratio[2] * lstm2)

with ProcessPoolExecutor(max_workers=2) as executor:
    for ratio, (hits, scores) in zip(
        ratios,
        tqdm(executor.map(search, linear_combinations), total=len(linear_combinations)),
    ):
        hits_per_method[tuple(ratio)] = hits
        scores_per_method[tuple(ratio)] = scores


# %%


def compute_is_correct(results: ndarray) -> ndarray:
    # queries -> levels -> hits
    return numpy.asarray(
        [
            (mapping_array[query] == mapping_array[result]).T
            for query, result in zip(range(len(results)), results)
        ]
    )


print("Checking hits for correctness")
is_correct_all_per_combination = {
    name: compute_is_correct(results) for name, results in hits_per_method.items()
}

# %%

mean_correct_combinations = numpy.asarray(
    [
        (*name, is_correct[:, 3, 0].mean())
        for name, is_correct in is_correct_all_per_combination.items()
    ]
)

numpy.savez(
    figures.joinpath("seqvec_layer.npz"),
    mean_correct_combinations=mean_correct_combinations,
)

# %%

plot_data = dict()
for cnn, lstm1, lstm2, mean_correct in mean_correct_combinations:
    plot_coordinates = (cnn * (granularity - 1), lstm1 * (granularity - 1))
    plot_data[plot_coordinates] = mean_correct

figure, tax = ternary.figure(scale=granularity - 1)
tax.heatmap(
    plot_data, style="hexagonal", cmap="viridis", cbarlabel="Normalized accuracy"
)
tax.boundary()
tax.clear_matplotlib_ticks()
tax.get_axes().axis("off")

tax.bottom_axis_label("CNN", offset=0.06)
tax.right_axis_label("LSTM1", offset=0.13)
tax.left_axis_label("LSTM2", offset=0.13)

# noinspection PyProtectedMember
tax._redraw_labels()

ticks = list(numpy.linspace(0, 1, 11))
tax.ticks(ticks=ticks, tick_formats="%.1f", offset=0.02, label="asdf")

plt.tight_layout()
endfig(figures, "seqvec_layer")

# %%
