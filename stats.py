from nilearn.glm.first_level import FirstLevelModel
from scipy import stats


def fit_glm(fmri_img, design_matrix, mask):
    fmri_glm = FirstLevelModel(minimize_memory=False, mask_img=mask, standardize=False)
    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrix)
    return fmri_glm


def pearson_correlation(x, y):
    result = stats.pearsonr(x, y)
    return {"pearson_coeff": result.statistic, "p_val": result.pvalue}
