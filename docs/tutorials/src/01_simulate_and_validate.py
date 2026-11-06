# %% [markdown]
# # 1. Simulate and validate a cohort
#
# This tutorial simulates a synthetic cohort from the pd_v2 generative model, validates it, checks it against the
# model's design envelope and writes a data card. No real data are used.

# %%
from symphony_dt.data import check_envelope, simulate_cohort
from symphony_dt.governance import data_card_markdown

cohort = simulate_cohort(120, seed=7)
ds = cohort.dataset
ds

# %% [markdown]
# The dataset is in long format: one row per participant, visit and endpoint. Time is whole days since baseline.

# %%
ds.observations.head()

# %%
ds.summary(min_cell=10)

# %% [markdown]
# The true global parameters are known, which is what makes simulation useful for checking a fit.

# %%
{k: round(v, 3) for k, v in list(cohort.truth().items())[:8]}

# %% [markdown]
# ## The validity envelope
#
# The envelope compares the dataset with the domain pd_v2 was designed for. Each departure is a coded warning.

# %%
check_envelope(ds).to_dict()

# %% [markdown]
# ## A data card
#
# The data card reports aggregate counts only, with small cells suppressed, and the dataset fingerprint.

# %%
print(data_card_markdown(ds, title="Synthetic cohort", source="symphony_dt.simulate_cohort, seed 7"))

# %% [markdown]
# ## Plot one trajectory

# %%
from symphony_dt.plots import plot_trajectory

plot_trajectory(ds, ds.participant_ids[3], endpoint="updrs3");
