
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
def generate_data(n=1000, seed=0, beta1=1.05, alpha1=0.4, alpha2=0.3,binary_treatment=True, binary_cutoff=3.5):
    """生成模拟数据"""
    np.random.seed(seed)
    age = np.random.normal(65, 5, n)
    sodium = age / 18 + np.random.normal(size=n)
    if binary_treatment:
        if binary_cutoff is None:
            binary_cutoff = sodium.mean()
        sodium = (sodium > binary_cutoff).astype(int)
    blood_pressure = beta1 * sodium + 2 * age + np.random.normal(size=n)
    proteinuria = alpha1 * sodium + alpha2 * blood_pressure + np.random.normal(size=n)
    hypertension = (blood_pressure >= 140).astype(int) # not used, but could beused for binary outcomes
    return pd.DataFrame({'blood_pressure': blood_pressure, 'sodium': sodium,'age': age, 'proteinuria': proteinuria})

def estimate_causal_effect(Xt, y, model=LinearRegression(), treatment_idx=0,regression_coef=False):
    # TODO 1: 完成estimate_causal_effect函数
    model.fit(Xt, y)
    if regression_coef:
        # TODO
        Xt_ = pd.DataFrame.copy(Xt)
        Xt_.drop_duplicates(inplace=True)
        Xt1 = pd.DataFrame.copy(Xt_)
        Xt0 = pd.DataFrame.copy(Xt_)
        Xt1[Xt.columns[treatment_idx]] = 1
        Xt0[Xt.columns[treatment_idx]] = 0

        return sum(model.predict(Xt1) - model.predict(Xt0)) / len(Xt0)
    else:
        Xt1 = pd.DataFrame.copy(Xt)
        Xt1[Xt.columns[treatment_idx]] = 1
        Xt0 = pd.DataFrame.copy(Xt)
        Xt0[Xt.columns[treatment_idx]] = 0
        # TODO
        return sum(model.predict(Xt1) - model.predict(Xt0)) / len(Xt0)

def estimate_causal_effect_with_GCOM(Xt, y, model1=LinearRegression(), model2=LinearRegression(), treatment_idx=0,regression_coef=False):
    Xt_1 = Xt[Xt[Xt.columns[treatment_idx]].isin([1])]
    y_1 = y[Xt[Xt.columns[treatment_idx]].isin([1])]
    Xt_0 = Xt[Xt[Xt.columns[treatment_idx]].isin([0])]
    y_2 = y[Xt[Xt.columns[treatment_idx]].isin([0])]
    # TODO 3: 使用GCOM(Grouped Conditional Outcome Modeling)进行估计
    model1.fit(Xt_1, y_1)
    model2.fit(Xt_0, y_2)
    if regression_coef:

        Xt_ = pd.DataFrame.copy(Xt)
        Xt_.drop_duplicates(inplace=True)
        Xt1 = pd.DataFrame.copy(Xt_)
        Xt0 = pd.DataFrame.copy(Xt_)
        Xt1[Xt.columns[treatment_idx]] = 1
        Xt0[Xt.columns[treatment_idx]] = 0

        return sum(model1.predict(Xt1) - model2.predict(Xt0)) / len(Xt0)
    else:
        Xt1 = pd.DataFrame.copy(Xt)
        Xt1[Xt.columns[treatment_idx]] = 1
        Xt0 = pd.DataFrame.copy(Xt)
        Xt0[Xt.columns[treatment_idx]] = 0

        return sum(model1.predict(Xt1) - model2.predict(Xt0)) / len(Xt0)

def estimate_causal_effect_with_X_Learner(Xt, y, model1=LinearRegression(), model2=LinearRegression(), model3=LinearRegression(), model4=LinearRegression(), treatment_idx=0,regression_coef=False, gx = 0.8):
    Xt_1 = Xt[Xt[Xt.columns[treatment_idx]].isin([1])]
    y_1 = y[Xt[Xt.columns[treatment_idx]].isin([1])]
    Xt_0 = Xt[Xt[Xt.columns[treatment_idx]].isin([0])]
    y_2 = y[Xt[Xt.columns[treatment_idx]].isin([0])]
    # TODO 3: 使用GCOM(Grouped Conditional Outcome Modeling)进行估计
    model1.fit(Xt_1, y_1)
    model2.fit(Xt_0, y_2)
    
    d1 = y_1 - model2.predict(Xt_1)
    d0 = model1.predict(Xt_0) - y_2

    model3.fit(Xt_1, d1)
    model4.fit(Xt_0, d0)


    rt =  gx * sum(model3.predict(Xt_1) + (1 - gx) *  model4.predict(Xt_1)) / len(Xt_1) - gx * sum(model3.predict(Xt_0) + (1 - gx) * model4.predict(Xt_0)) / len(Xt_0)

    return rt



if __name__ == '__main__':
    binary_t_df = generate_data(beta1=1.05, alpha1=.4, alpha2=.3,binary_treatment=True, n=10000000)
    continuous_t_df = generate_data(beta1=1.05, alpha1=.4, alpha2=.3,binary_treatment=False, n=10000000)
    ate_est_naive = None
    ate_est_adjust_all = None
    ate_est_adjust_age = None

    for df, name in zip([binary_t_df, continuous_t_df],['Binary Treatment Data', 'Continuous Treatment Data']):
        print()
        print('### {} ###'.format(name))
        print()

        # Adjustment formula estimates
        ate_est_naive = estimate_causal_effect(df[['sodium']],df['blood_pressure'], treatment_idx=0)
        ate_est_adjust_all = estimate_causal_effect(df[['sodium', 'age','proteinuria']],df['blood_pressure'],treatment_idx=0)
        # TODO 2: 仅以年龄作为调整集进行估计
        ate_est_adjust_age = estimate_causal_effect(df[['sodium','age']],df['blood_pressure'],treatment_idx=0)
        print('# Adjustment Formula Estimates #')
        print('Naive ATE estimate:\t\t\t\t\t\t\t', ate_est_naive)
        print('ATE estimate adjusting for all covariates:\t',ate_est_adjust_all)
        print('ATE estimate adjusting for age:\t\t\t\t', ate_est_adjust_age)
        print()

        # Linear regression coefficient estimates
        ate_est_naive = estimate_causal_effect(df[['sodium']],df['blood_pressure'], treatment_idx=0,regression_coef=True)
        ate_est_adjust_all = estimate_causal_effect(df[['sodium', 'age','proteinuria']],df['blood_pressure'],treatment_idx=0,regression_coef=True)

        # TODO 2: 仅以年龄作为调整集进行估计
        ate_est_adjust_age = estimate_causal_effect(df[['sodium', 'age']],df['blood_pressure'],treatment_idx=0,regression_coef=True)
        print('# Regression Coefficient Estimates #')
        print('Naive ATE estimate:\t\t\t\t\t\t\t', ate_est_naive)
        print('ATE estimate adjusting for all covariates:\t',ate_est_adjust_all)
        print('ATE estimate adjusting for age:\t\t\t\t', ate_est_adjust_age)
        print()

    # Adjustment formula estimates with GCOM
    ate_est_naive = estimate_causal_effect_with_GCOM(binary_t_df[['sodium']],binary_t_df['blood_pressure'], treatment_idx=0)
    ate_est_adjust_all = estimate_causal_effect_with_GCOM(binary_t_df[['sodium', 'age','proteinuria']],binary_t_df['blood_pressure'],treatment_idx=0)
    # TODO 2: 仅以年龄作为调整集进行估计
    ate_est_adjust_age = estimate_causal_effect_with_GCOM(binary_t_df[['sodium','age']],binary_t_df['blood_pressure'],treatment_idx=0)
    print('# Adjustment Formula Estimates #')
    print('Naive ATE estimate:\t\t\t\t\t\t\t', ate_est_naive)
    print('ATE estimate adjusting for all covariates:\t',ate_est_adjust_all)
    print('ATE estimate adjusting for age:\t\t\t\t', ate_est_adjust_age)
    print()


    # X-Learner
    ate_est_adjust_age = estimate_causal_effect_with_X_Learner(binary_t_df[['sodium','age']],binary_t_df['blood_pressure'],treatment_idx=0)
    print('# Adjustment Formula Estimates #')
    print('Naive ATE estimate:\t\t\t\t\t\t\t', ate_est_adjust_age)