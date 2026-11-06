"""Stub of the study's covariate manifest: the declared model inputs of the synthetic case."""

PATHWAY = ["pathway_1", "pathway_2"]


def model_inputs():
    return {
        "pathway_A_z": list(PATHWAY),
        "demographic_Gamma_d": ["age_baseline_years", "sex", "genotype_assay_available"],
    }


def covariates(level):
    return {"age_baseline_years", "sex", "genotype_assay_available", *PATHWAY}
