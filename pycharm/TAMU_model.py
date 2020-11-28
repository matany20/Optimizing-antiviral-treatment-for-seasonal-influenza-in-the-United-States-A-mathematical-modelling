import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import itertools
import pickle
from matplotlib import pyplot as plt
import datetime

#######################
# --- Set indices --- #
#######################

# Age groups
A = {0: '0-4', 1: '5-19', 2: '20-49', 3: '50-64', 4: '65+'}

# Population groups
P = {0: 'High', 1: 'Low'}

# All combination:
N = {i: group for i, group in enumerate(itertools.product(P.values(), A.values()))}

########################
# -- Set Parameters -- #
########################

# Load Parameters
with open('../model_data/parameters.pickle', 'rb') as pickle_in:
    parameters = pickle.load(pickle_in)

# Transmissibility (beta_jk - for each age group, for symptomatic/asymptomatic)
# beta = parameters['beta']

# Demography:
with open('../model_data/demography.pickle', 'rb') as pickle_in:
    demography = pickle.load(pickle_in)

# hospitalizations data for high risk with/without treatment and low risk:
with open('../model_data/hospitalizations.pickle', 'rb') as pickle_in:
    hospitalizations = pickle.load(pickle_in)

# Treatment proportion for each age group:
with open('../model_data/treatment_prop.pickle', 'rb') as pickle_in:
    treatment_prop = pickle.load(pickle_in)

# duration of infection (days)
varphi = parameters['varphi']

# Asymptomatic fraction
f = parameters['f']

# Contact matrix
C = pd.read_pickle('../model_data/contact_matrix_adj.pickle').values

# Cosine phase - estimated
# phi = parameters['phi']

# Vaccination coverage for each subgroup
vaccination_coverage = pd.read_pickle('../model_data/vaccination_coverage.pickle')

# Vaccination efficacy
eta = parameters['eta']

# Population size
population_size = pd.read_pickle('../model_data/population_data.pickle')

# Epsilon (small noise) - only for non-zero population groups
eps = 1e-8

# Load baseline policy treatment
with open('../model_data/policy.pickle', 'rb') as pickle_in:
    base_policy = pickle.load(pickle_in)

# Load immunity proportions
with open('../model_data/immunity.pickle', 'rb') as pickle_in:
    immunity = pickle.load(pickle_in)

# Load log viral load
with open('../model_data/viral_load.pickle', 'rb') as pickle_in:
    VL = pickle.load(pickle_in)


#######################
# ---- Run Model ---- #
#######################

def run_model(beta, phi, init_I, init_A, population_size=population_size, policy=base_policy,
              C=C, vaccination_coverage=vaccination_coverage, eps=eps, varphi=varphi, VL=VL):
    """Receives all model's data and parameters, runs it for a season and returns the results"""

    # Initialize lists to save the states throughout the time steps
    S, i_s, I, a, A, R = [], [], [], [], [], []

    # Initialize a list for the newly infected
    new_I, new_A = [], []

    # Initialize a list fot the lambdas
    L = []

    # Run the model
    for t in range(365):
        # If first iteration - initialize all states
        if t % 365 == 0:
            # Initialize S_0 to population size of each age-group
            S.append(population_size.copy())
            # Initialize R - with only the naturally immune individuals
            R.append(np.zeros(len(N)))
            R[-1][0:10:5] = immunity['0-4'] * population_size[0:10:5]
            R[-1][1:10:5] = immunity['5-19'] * population_size[1:10:5]
            R[-1][2:10:5] = immunity['20-49'] * population_size[2:10:5]
            R[-1][3:10:5] = immunity['50-64'] * population_size[3:10:5]
            R[-1][4:10:5] = immunity['65+'] * population_size[4:10:5]
            # subtract R0 from S0:
            S[-1] -= R[-1]
            # Calculates the effectively vaccinated population, add it to R0 and subtract them from S0
            vaccinated_population = S[-1] * eta * vaccination_coverage
            R[-1] += vaccinated_population
            S[-1] -= vaccinated_population

            # Initialize I (symptomatic) to 5*10**-4 ????
            i_s.append(np.zeros((len(N), varphi)))
            i_s[-1][:, 0] = init_I * population_size
            I.append(i_s[-1].sum(axis=1))

            # Initialize A (asymptomatic) to 5*0.5*10**-4 ????
            a.append(np.zeros((len(N), varphi)))
            a[-1][:, 0] = init_A * population_size
            A.append(a[-1].sum(axis=1))

            # Subtract I_0 and A_0 from S_0
            S[-1] -= (I[-1] + A[-1])

            # Zero newly infected on the first day of the season
            new_I.append(np.zeros(len(N)))
            new_A.append(np.zeros(len(N)))
        # Not a new season
        # Calculate lambda (High risk symptomatic + Low risk symptomatic + Asymptomatic
        lambda_t = eps + (beta * (C.T.dot((i_s[-1][0:5, :] * policy['without'][:varphi] * VL['high'][:varphi] + \
                                           i_s[-1][0:5, :] * policy['day1'][:varphi] * VL['day1'][:varphi] + \
                                           i_s[-1][0:5, :] * policy['day2'][:varphi] * VL['day23'][:varphi] + \
                                           i_s[-1][0:5, :] * policy['day3'][:varphi] * VL['day23'][:varphi]).sum(axis=1) + \
                                          (i_s[-1][5:10, :] * VL['low'][:varphi]).sum(axis=1) + \
                                          ((a[-1][0:5, :] + a[-1][5:10, :]) * (VL['asymptomatic'][:varphi])).sum(axis=1))) * (
                                  np.maximum(0,1 + np.cos((2 * np.pi * (t - phi)) / 365))))
        L.append(lambda_t)
        lambda_t = np.tile(lambda_t, 2)  # fitting lambda_t size to (10X1)
        # R(t)
        R.append(R[-1] + i_s[-1][:, -1] + a[-1][:, -1])

        # I(t)
        # Calculate new i matrix for day t
        i_day = np.zeros((len(N), varphi))
        i_day[:, 0] = S[-1] * lambda_t * (1 - f)
        i_day[:, 1:] = i_s[-1][:, :-1]
        i_s.append(i_day)
        I.append(i_s[-1].sum(axis=1))

        # A(t)
        # Calculate new a matrix for day t
        a_day = np.zeros((len(N), varphi))
        a_day[:, 0] = S[-1] * lambda_t * f
        a_day[:, 1:] = a[-1][:, :-1]
        a.append(a_day)
        A.append(a[-1].sum(axis=1))

        # S(t)
        S.append(S[-1] - i_s[-1][:, 0] - a[-1][:, 0])

        # Save current new_I and new_A
        new_I.append(i_s[-1][:, 0])
        new_A.append(a[-1][:, 0])

    # Return the model results
    return {'S': S, 'i': i_s, 'I': I, 'a': a, 'A': A, 'R': R, 'new_I': new_I, 'new_A': new_A, 'L': L}


#######################
# ------ Plots ------ #
#######################

def plot_compartments_spec(model_results, age_group, plot_s=False, plot_i=True, plot_a=False, plot_R=False):
    "The function gets model result and plot model compartments S,A,I,R for a given age-group index"

    # Preparing model results data for plot
    S = np.array(model_results["S"])
    R = np.array(model_results["R"])
    I = np.array(model_results["I"])
    A = np.array(model_results["A"])
    plot_dict = {"S": S[:, age_group].sum(axis=1), "I": I[:, age_group].sum(axis=1), "A": A[:, age_group].sum(axis=1), \
                 "R": R[:, age_group].sum(axis=1)}  # dict in order to convert to df for plot

    # convert df for plotting
    plot_df = pd.DataFrame.from_dict(plot_dict)

    # plotting
    plot_df.plot(figsize=(12, 4), grid=True)
    plt.show()


def plot_calibrated_model_two(data, mdl_data, with_shifting):
    """ The function gets the results of the model and plot for each age group the model results and the data"""
    mrg_mdl_dt = mdl_data[['0-4_tot', '5-19_tot', '20-49_tot', '50-64_tot', '65+_tot']].merge(data, left_index=True,
                                                                                              right_on='dates').merge(
        with_shifting[['0-4_tot', '5-19_tot', '20-49_tot', '50-64_tot', '65+_tot']], left_on='dates', right_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, groups in zip(axes.flat, range(5)):
        mrg_mdl_dt.plot(x='dates', y=[A[groups] + '_tot_x', A[groups], A[groups] + '_tot_y'], style=['-', '.', '-'],
                        ax=ax)
    print('MSE:', MSE(data, mdl_data))


def plot_calibrated_model(data, mdl_data):
    """ The function gets the results of the model and plot for each age group the model results and the data"""
    mrg_mdl_dt = mdl_data[['0-4_tot', '5-19_tot', '20-49_tot', '50-64_tot', '65+_tot']].merge(data, left_index=True,
                                                                                              right_on='dates')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, groups in zip(axes.flat, range(5)):
        mrg_mdl_dt.plot(x='dates', y=[A[groups] + '_tot', A[groups]], style=['-', '.'], ax=ax)
    print('MSE:', MSE(data, mdl_data))
    plt.show()
    plt.close()


def plot_fitting_model_data_total(model_dt, data, state, population_amount):
    """ Plotting the calibrated model vs data over the seasons """
    plt.scatter(data.index, data['total'] * population_amount[state], label='Data', color='#012C61', s=8, zorder=2)
    plt.plot(model_dt.index, model_dt['total'] * population_amount[state], label='Model', color='#FFB700',
             linewidth=2.5, zorder=1)
    plt.legend()
    plt.title(state, fontweight='bold')
    plt.ylabel('Weekly Influenza cases', fontweight='bold')
    plt.yticks([0, 50000, 100000, 150000, 200000], fontweight='bold')


def plot_fitted_attack_rate_bars(model_AR, data_AR, state):
    """ Plotting the mean attack rate of the calibrated model vs data  """
    # set width of bar
    barWidth = 0.25

    # set height of bar
    bars_model = [model_AR[A[i]] for i in range(5)]
    bars_data = [data_AR[A[i]] for i in range(5)]

    # Set position of bar on X axis
    r1 = np.arange(len(bars_model))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, bars_model, color='#FFB700', width=barWidth, edgecolor='white', label='Model')
    plt.bar(r2, bars_data, color='#012C61', width=barWidth, edgecolor='white', label='Data')

    # Add xticks + yticks on the middle of the group bars
    plt.xlabel('Age Group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars_model))], ['0-4y', '5-19y', '20-49y', '50-64y', '60+y'],
               fontweight='bold')
    plt.ylabel('Proportion of Influenza cases', fontweight='bold')
    plt.ylim(0, 0.5)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], fontweight='bold')

    # Create legend & Show graphic
    plt.legend()
    plt.title(state, fontweight='bold')


###############################
# ------ Model Fitting ------ #
###############################

def get_state_data(state):
    """Function gets state name and returns its data by season as dict. """
    with open('./model_data/{}.pickle'.format(state + '_data'), 'rb') as pickle_file:
        state_data = pickle.load(pickle_file)
    return {season: (state_data[state_data.season == season]).copy() for season in range(2015, 2020)}


def data_with_dates(data, dates):
    """ The function gets data and dates list and adding for each week in the data its date"""
    # Add date for each week in dates
    dates_data = pd.DataFrame(index=dates).resample('W').sum()[:-1]
    dates_data['week'] = dates_data.index.week
    dates_data['dates'] = dates_data.index

    # Get names of indexes for which column Age has value 30
    indexNames = dates_data[dates_data['week'] == 53].index

    # Delete these row indexes from dataFrame
    dates_data.drop(indexNames, inplace=True)

    # Add weeks' date for each week in the data
    relevant_data = data.merge(dates_data, right_on='week', left_on='WEEK', how='right')
    # order the data in ascending order by date
    relevant_data.sort_values('dates', ascending=True, inplace=True)
    relevant_data['total'] = relevant_data['total'].fillna(0)
    return relevant_data


def model_infected_symp_cases(model_results, dates_list, season):
    """ The function gets model results, dates list and season, returns DataFrame of the model new symptomatic cases
     by week in the season"""

    # Get new symptomatic infected without the initials (0)
    new_I = np.array(model_results['new_I'])[1:, :]

    # Creating new cases DataFrame:
    model_df = pd.DataFrame(new_I, columns=[N[k] for k in N.keys()], index=np.array(dates_list))
    # Calculating number of cases by week
    model_df = model_df.resample('W').sum()

    # Adding the total cases as column (Low risk, High risk) for each age group:
    for i, age_group in [(0, [0, 5]), (1, [1, 6]), (2, [2, 7]), (3, [3, 8]), (4, [4, 9])]:
        model_df[A[i] + '_tot'] = model_df.values[:, age_group].sum(axis=1)

    # Add the total cases
    model_df['total'] = model_df.apply(lambda x: (
        x[['0-4_tot', '5-19_tot', '20-49_tot', '50-64_tot', '65+_tot']].values).sum(), axis=1)

    if season == 2016 :
        model_df = model_df[:-1].copy()

    # Returning only data until the week 39 of season year (as in the the CDC data)
    return model_df[((model_df.index.week <= 39) | (model_df.index.week == 52) )| (model_df.index.year == season - 1)]



def model_attack_rate(model_data, state):
    """ The function gets model results and returns the mean attack rate for each age group"""

    # Add season column to the model results
    model_data['dates'] = model_data.index
    model_data['season'] = model_data.apply(lambda s: s.dates.year if s.dates.week <= 39 else s.dates.year + 1, axis=1)

    # Calculate sum weekly cases
    # model_data = model_data.groupby('season', as_index=False).sum()
    model_data = model_data.sum(axis=0)

    # Get mean cases by season
    # model_data = model_data.mean(axis=0)
    return {A[age]: model_data[A[age] + '_tot'] / demography[state][age] for age in range(5)}


def get_data_attack_rate(data, state):
    """ The function gets the data and returns the mean attack rate for each age group"""

    # Calculate sum weekly cases
    # data = data.groupby('season', as_index=False).sum()
    data = data.sum(axis=0)

    # Get mean cases by season
    # data = data.mean(axis=0)
    return {A[age]: data[A[age]] / demography[state][age] for age in range(5)}


def get_data_proportion(data):
    """ The function gets the data and returns the proportion  for each age group"""

    # Calculate sum weekly cases
    # data = data.groupby('season', as_index=False).sum()
    data = data.sum(axis=0)

    # Get mean cases by season
    # data = data.mean(axis=0)
    return {A[age]: data[A[age]] / data['total'] for age in range(5)}


def model_proportion(model_data):
    """ The function gets model results and returns the mean attack rate for each age group"""

    # Add season column to the model results
    model_data['dates'] = model_data.index
    model_data['season'] = model_data.apply(lambda s: s.dates.year if s.dates.week <= 39 else s.dates.year + 1, axis=1)

    # Calculate sum weekly cases
    # model_data = model_data.groupby('season', as_index=False).sum()
    model_data = model_data.sum(axis=0)

    # Get mean cases by season
    # model_data = model_data.mean(axis=0)
    return {A[age]: model_data[A[age] + '_tot'] / model_data['total'] for age in range(5)}

##################################
# ------ Helper Functions ------ #
##################################

def get_date_from_week(week_num, year):
    """The function gets year and week number, returns the dates list starting from the week and year given"""
    # calculate the starting date of the week
    d = datetime.datetime.strptime(str(year) + '-W' + str(week_num) + '-1', "%Y-W%W-%w")

    if weeks_for_year(year) == 53:
        full_year = 365+7
    else:
        full_year = 365

    return [pd.Timestamp(d) + pd.Timedelta(i, unit='d') for i in range(full_year)]


def MSE(data, model_data_processed):
    mse = 0
    for i in range(5):
        mse += np.mean((data[A[i]].values - model_data_processed[A[i] + '_tot'].values) ** 2)
    return mse


def create_new_policy_matrices(new_treated_1d, new_treated_2d, new_treated_3d):


    # treatment policy day1:
    treat_day1 = np.zeros((len(A), varphi))
    treat_day1[0, 3:] = new_treated_1d[0]
    treat_day1[1, 3:] = new_treated_1d[1]
    treat_day1[2, 3:] = new_treated_1d[2]
    treat_day1[3, 3:] = new_treated_1d[3]
    treat_day1[4, 3:] = new_treated_1d[4]

    # treatment policy day2:
    treat_day2 = np.zeros((len(A), varphi))
    treat_day2[0, 4:] = new_treated_2d[0]
    treat_day2[1, 4:] = new_treated_2d[1]
    treat_day2[2, 4:] = new_treated_2d[2]
    treat_day2[3, 4:] = new_treated_2d[3]
    treat_day2[4, 4:] = new_treated_2d[4]

    # treatment policy day3:
    treat_day3 = np.zeros((len(A), varphi))
    treat_day3[0, 5:] = new_treated_3d[0]
    treat_day3[1, 5:] = new_treated_3d[1]
    treat_day3[2, 5:] = new_treated_3d[2]
    treat_day3[3, 5:] = new_treated_3d[3]
    treat_day3[4, 5:] = new_treated_3d[4]

    # without treatment:
    treat_without = np.ones((len(A), varphi)) - treat_day1 - treat_day2 - treat_day3

    # dictionary of policy
    return {'without': treat_without, 'day1': treat_day1, 'day2': treat_day2, 'day3': treat_day3}


def shifting_treated_within_2d(state, seasons, fitted_params, vacc_coverage=None):
    # proportion of getting treated within 48h given got treatment
    treated_within_2d_prop = 0.545

    # total treated among high risk at each age group
    total_treatment_age_group = np.array([treatment_prop[A[i]] / treated_within_2d_prop for i in range(5)])

    # proportion of treated among high risk within 2(48h),3(72h),4+(96+h) days after symptoms on set
    treated_2d_baseline = np.array([treatment_prop[A[i]] * 0.5 for i in range(5)])
    treated_3d_baseline = treated_2d_baseline.copy()
    treated_after_2d_baseline = total_treatment_age_group - treated_2d_baseline
    treated_4d_baseline = treated_after_2d_baseline - treated_3d_baseline

    # Pandas DFs that  holds the results for cases and hospitalizations each point: diff. from baseline
    columns = ['season', 'new_policy_cases']
    results_df = pd.DataFrame(columns=columns)

    columns_hos = ['season', 'new_policy_hospital']
    hos_results_df = pd.DataFrame(columns=columns_hos)

    for season in seasons:
        result_to_df, hosp_result_to_df = {}, {}
        result_to_df['season'] = season
        hosp_result_to_df['season'] = season
        fitted_beta = fitted_params[str(season)]['beta']
        fitted_phi = fitted_params[str(season)]['phi']

        # Get results for each new policy
        for treatment_percent in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            # Calculate the new treatment policy percentage:
            shifted_percent = treated_after_2d_baseline * treatment_percent
            new_treated_2d = np.minimum(treated_2d_baseline + shifted_percent, np.ones(len(A)))
            new_treated_4d = np.minimum(shifted_percent, treated_4d_baseline)
            new_treated_3d = np.maximum(
                treated_3d_baseline - np.maximum(shifted_percent - new_treated_4d, np.zeros(len(A))), 0)

            new_policy = create_new_policy_matrices(np.zeros(len(A)), new_treated_2d, new_treated_3d)
            people_shifted = (shifted_percent * demography[state] * np.array([0.052, 0.106, 0.149, 0.33, 0.512])).sum()
            # creating a result dict that wi'll be added to the result df as a row
            result_to_df['percentage_change'] = treatment_percent
            result_to_df['shifted'] = people_shifted
            hosp_result_to_df['percentage_change'] = treatment_percent
            hosp_result_to_df['shifted'] = people_shifted

            model_results = run_model(fitted_beta, fitted_phi * 10 ** (4), 1e-4, 1e-4,
                                      population_size=population_size[state],
                                      vaccination_coverage= vacc_coverage,
                                      policy=new_policy)

            new_I = np.array(model_results['new_I'])

            # Calculate the total number of new cases (symptomatic)
            total_cases = new_I.sum()

            # Calculate the total number of new hospitalizations (symptomatic) low risk + high risk w\o treatment + high
            # with treatment.
            total_hosp = (new_I[:, 5:].sum(axis=0) * hospitalizations['Low']).sum() + \
                         (new_I[:, :5].sum(axis=0) * (1 - (new_treated_2d + new_treated_3d)) * \
                          hospitalizations['High']).sum() + \
                         (new_I[:, :5].sum(axis=0) * (new_treated_2d + new_treated_3d) *
                          hospitalizations['High_treated']).sum()

            # If we're at the baseline case, save it as the baseline case
            if treatment_percent == 0:
                baseline_cases = total_cases
                baseline_hospitalization = total_hosp

            # Save as new policy
            result_to_df['new_policy_cases'] = total_cases
            hosp_result_to_df['new_policy_hospital'] = total_hosp

            # Add baseline cases
            result_to_df['baseline_cases'] = baseline_cases
            hosp_result_to_df['baseline_hospital'] = baseline_hospitalization

            # Insert result as a new row for the results DF
            results_df = results_df.append(result_to_df, ignore_index=True)
            hos_results_df = hos_results_df.append(hosp_result_to_df, ignore_index=True)

    return {'Cases': results_df, 'Hospital': hos_results_df}


def model_simulations_without_treatment(state, seasons, fitted_params, vacc_coverage=None):

    # Pandas DFs that  holds the results for cases and hospitalizations each point: diff. from baseline
    columns = ['season', 'new_policy']
    results_df = pd.DataFrame(columns=columns)

    hos_results_df = pd.DataFrame(columns=columns)

    for season in seasons:
        result_to_df, hosp_result_to_df = {}, {}
        result_to_df['season'] = season
        hosp_result_to_df['season'] = season
        fitted_beta = fitted_params[str(season)]['beta']
        fitted_phi = fitted_params[str(season)]['phi']


        new_policy = create_new_policy_matrices(np.zeros(len(A)), np.zeros(len(A)), np.zeros(len(A)))

        model_results = run_model(fitted_beta, fitted_phi * 10 ** (4), 1e-4, 1e-4,
                                  population_size=population_size[state],
                                  vaccination_coverage= vacc_coverage,
                                  policy=new_policy)

        new_I = np.array(model_results['new_I'])

        # Calculate the total number of new cases (symptomatic)
        total_cases = new_I.sum()

        # Calculate the total number of new hospitalizations (symptomatic) low risk + high risk w\o treatment + high
        # with treatment.
        total_hosp = (new_I[:, 5:].sum(axis=0) * hospitalizations['Low']).sum() + \
                     (new_I[:, :5].sum(axis=0) * hospitalizations['High']).sum()



        # Save as new policy
        result_to_df['new_policy'] = total_cases
        hosp_result_to_df['new_policy'] = total_hosp


        # Insert result as a new row for the results DF
        results_df = results_df.append(result_to_df, ignore_index=True)
        hos_results_df = hos_results_df.append(hosp_result_to_df, ignore_index=True)
    results_df['world'] = 'cases'
    hos_results_df['world'] = 'hosp'
    return pd.concat([results_df,hos_results_df])



def shifting_treated_within_2d_hosp_sensetivity(state, seasons, fitted_params, reduction_child, reduction_adult, new_VL, vacc_coverage=None):
    # proportion of getting treated within 48h given got treatment
    treated_within_2d_prop = 0.545

    # total treated among high risk at each age group
    total_treatment_age_group = np.array([treatment_prop[A[i]] / treated_within_2d_prop for i in range(5)])

    # proportion of treated among high risk within 2(48h),3(72h),4+(96+h) days after symptoms on set
    treated_2d_baseline = np.array([treatment_prop[A[i]] * 0.5 for i in range(5)])
    treated_3d_baseline = treated_2d_baseline.copy()
    treated_after_2d_baseline = total_treatment_age_group - treated_2d_baseline
    treated_4d_baseline = treated_after_2d_baseline - treated_3d_baseline

    # Pandas DFs that  holds the results for
    # hospitalizations each point: diff. from baseline
    columns_hos = ['season', 'new_policy_hospital']
    hos_results_df = pd.DataFrame(columns=columns_hos)

    # calculating hospitalization probability for high risk
    treatment_reduction = np.zeros(len(A))
    # calculating new probability for <=19y
    treatment_reduction[:2] = np.array(hospitalizations['High'][:2]) * (1 - reduction_child)
    treatment_reduction[2:] = np.array(hospitalizations['High'][2:]) * (1 - reduction_adult)

    for season in seasons:
        hosp_result_to_df = {}
        hosp_result_to_df['season'] = season
        fitted_beta = fitted_params[str(season)]['beta']
        fitted_phi = fitted_params[str(season)]['phi']
        hosp_result_to_df['reduction_kids'] = reduction_child
        hosp_result_to_df['reduction_adults'] = reduction_adult
        # Get results for each new policy
        for treatment_percent in [0,1]:
            # Calculate the new treatment policy percentage:
            shifted_percent = treated_after_2d_baseline * treatment_percent
            new_treated_2d = np.minimum(treated_2d_baseline + shifted_percent, np.ones(len(A)))
            new_treated_4d = np.minimum(shifted_percent, treated_4d_baseline)
            new_treated_3d = np.maximum(
                treated_3d_baseline - np.maximum(shifted_percent - new_treated_4d, np.zeros(len(A))), 0)

            new_policy = create_new_policy_matrices(np.zeros(len(A)), new_treated_2d, new_treated_3d)
            people_shifted = (shifted_percent * demography[state] * np.array([0.052, 0.106, 0.149, 0.33, 0.512])).sum()
            # creating a result dict that wi'll be added to the result df as a row

            hosp_result_to_df['percentage_change'] = treatment_percent
            hosp_result_to_df['shifted'] = people_shifted

            model_results = run_model(fitted_beta, fitted_phi * 10 ** (4), 1e-4, 1e-4,
                                      population_size=population_size[state],
                                      vaccination_coverage= vacc_coverage,
                                      policy=new_policy, VL=new_VL)

            new_I = np.array(model_results['new_I'])


            # Calculate the total number of new hospitalizations (symptomatic) low risk + high risk w\o treatment + high
            # with treatment.
            total_hosp = (new_I[:, 5:].sum(axis=0) * hospitalizations['Low']).sum() + \
                         (new_I[:, :5].sum(axis=0) * (1 - (new_treated_2d + new_treated_3d)) * \
                          hospitalizations['High']).sum() + \
                         (new_I[:, :5].sum(axis=0) * (new_treated_2d + new_treated_3d) *
                          treatment_reduction).sum()

            # If we're at the baseline case, save it as the baseline case
            if treatment_percent == 0:
                baseline_hospitalization = total_hosp

            hosp_result_to_df['new_policy_hospital'] = total_hosp

            # Add baseline cases
            hosp_result_to_df['baseline_hospital'] = baseline_hospitalization

            # Insert result as a new row for the results DF
            hos_results_df = hos_results_df.append(hosp_result_to_df, ignore_index=True)

    return  hos_results_df[hos_results_df.percentage_change != 0]


def increase_treated(state, seasons, fitted_params, treatment_increase, vacc_coverage, baseline,
                     vacc_increase=[0.8, 0.9, 1, 1.1, 1.2], policy_factor = 1, reduction_kids=None,
                     reduction_adults=None, new_VL=None):

    # treated_within_2d_prop = 0.545

    # array of treatment effectively:
    effectively_treated = np.array([treatment_prop[A[i]] for i in range(len(A))])
    # total_treated_by_age_effect = effectively_treated / treated_within_2d_prop
    high_risk_by_age = np.array([0.052, 0.106, 0.149, 0.33, 0.512])
    high_risk_weights = high_risk_by_age / high_risk_by_age.sum()
    total_treated = (effectively_treated * high_risk_weights).sum()
    # Pandas DF that  holds the results for each point: diff. from baseline
    columns = ['percentage_increase', 'vacc_coverage', 'season', 'baseline_cases', 'new_policy_cases', 'shifted']
    results_df = pd.DataFrame(columns=columns)

    columns_hos = ['percentage_increase', 'vacc_coverage', 'season', 'baseline_hospital', 'new_policy_hospital',
                   'shifted']
    hos_results_df = pd.DataFrame(columns=columns_hos)

    if reduction_kids is not None and reduction_adults is not None:
        # calculating hospitalization probability for high risk
        treatment_reduction = np.zeros(len(A))
        # calculating new probability for <=19y
        treatment_reduction[:2] = np.array(hospitalizations['High'][:2]) * (1 - reduction_kids)
        treatment_reduction[2:] = np.array(hospitalizations['High'][2:]) * (1 - reduction_adults)
        hospitalizations['High_treated'] = treatment_reduction

    # Get results for each new policy
    for treatment_percent in treatment_increase:
        new_treated_2d = np.array([treatment_percent] * len(A)) * policy_factor
        new_treated_3d = np.array([treatment_percent] * len(A)) * (1 - policy_factor)

        new_policy = create_new_policy_matrices(np.zeros(len(A)), new_treated_2d, new_treated_3d)
        # creating a result dict that wi'll be added to the result df as a row
        result_to_df = {'percentage_increase': treatment_percent, 'shifted': treatment_percent - total_treated}
        hosp_result_to_df = {'percentage_increase': treatment_percent,
                             'shifted': treatment_percent - total_treated}

        if reduction_kids is not None and reduction_adults is not None:
            hosp_result_to_df['reduction_kids'] = reduction_kids
            hosp_result_to_df['reduction_adults'] = reduction_adults

        # run the model with different seasons for all the seasons, taking the average as final result
        for season in seasons:
            result_to_df['season'] = season
            hosp_result_to_df['season'] = season
            fitted_beta = fitted_params[str(season)]['beta']
            fitted_phi = fitted_params[str(season)]['phi']

            # run the model among the different vaccination coverage
            for vacc in vacc_increase:
                result_to_df['vacc_coverage'] = vacc
                hosp_result_to_df['vacc_coverage'] = vacc

                # run the model with the calibrated parameter for the base
                if new_VL is None:
                    model_results = run_model(fitted_beta, fitted_phi * 10 ** (4), 1e-4, 1e-4,
                                              population_size=population_size[state],
                                              vaccination_coverage=np.minimum(vacc_coverage * vacc, 1.0),
                                              policy=new_policy)
                else:
                    model_results = run_model(fitted_beta, fitted_phi * 10 ** (4), 1e-4, 1e-4,
                                              population_size=population_size[state],
                                              vaccination_coverage=np.minimum(vacc_coverage * vacc, 1.0),
                                              policy=new_policy, VL=new_VL)

                new_I = np.array(model_results['new_I'])

                # Calculate the total number of new cases (symptomatic)
                total_cases = new_I.sum()

            # Calculate the total number of new hospitalizations (symptomatic) low risk + high risk w\o treatment + high
                # with treatment.
                total_hosp = (new_I[:, 5:].sum(axis=0) * hospitalizations['Low']).sum() + \
                             (new_I[:, :5].sum(axis=0) * (1 - (new_treated_2d + new_treated_3d)) * \
                              hospitalizations['High']).sum() + \
                             (new_I[:, :5].sum(axis=0) * (new_treated_2d + new_treated_3d) *
                              hospitalizations['High_treated']).sum()

                # Save as new policy
                result_to_df['new_policy_cases'] = total_cases
                hosp_result_to_df['new_policy_hospital'] = total_hosp

                result_to_df['baseline_cases'] = baseline[vacc][str(season)]['Cases'].sum()
                hosp_result_to_df['baseline_hospital'] = baseline[vacc][str(season)]['Hospital'].sum()
                # Insert result as a new row for the results DF
                results_df = results_df.append(result_to_df, ignore_index=True)
                hos_results_df = hos_results_df.append(hosp_result_to_df, ignore_index=True)

    return {'Cases': results_df, 'Hospital': hos_results_df}


def get_baseline_results_age_group(state, seasons, fitted_parameters, vacc_coverage = None, reduction_kids=None,
                                   reduction_adults=None, new_VL=None):

    # results dictionary
    baseline_season = {}

    # array of treatment effectively:
    effectively_treated = np.array([treatment_prop[A[i]] for i in range(len(A))])
    if reduction_kids is not None and reduction_adults is not None:
        # calculating hospitalization probability for high risk
        treatment_reduction = np.zeros(len(A))
        # calculating new probability for <=19y
        treatment_reduction[:2] = np.array(hospitalizations['High'][:2]) * (1 - reduction_kids)
        treatment_reduction[2:] = np.array(hospitalizations['High'][2:]) * (1 - reduction_adults)
        hospitalizations['High_treated'] = treatment_reduction

    # run over all season:
    for season in seasons:

        # obtain season's parameters
        beta = fitted_parameters[str(season)]['beta']
        phi = fitted_parameters[str(season)]['phi']
        if new_VL is None:
            model_results = run_model(beta, phi * 10 ** (4), 1e-4, 1e-4,
                                      population_size=population_size[state],
                                      vaccination_coverage=vacc_coverage)
        else:
            model_results = run_model(beta, phi * 10 ** (4), 1e-4, 1e-4,
                                      population_size=population_size[state],
                                      vaccination_coverage=vacc_coverage, VL=new_VL)
        new_I = np.array(model_results['new_I']).sum(axis = 0)

        # calculating total symptomatic cases and per age group
        cases = np.array([new_I[[i, i+5]].sum() for i in A.keys()])
        hospital = (new_I[5:] * hospitalizations['Low']) + \
                   (new_I[:5] * (1 - effectively_treated) * hospitalizations['High']) + \
                   (new_I[:5] * effectively_treated * hospitalizations['High_treated'])

        # insert baseline into the dictionary result:
        baseline_season[str(season)] = {'Cases': cases, "Hospital": hospital}

    return baseline_season


def averted_by_age_group(state, seasons, fitted_parameters, baseline, age_idx, vacc_coverage, prop_2d= 0.5,
                         policy_2d= False):

    treated_within_2d_prop = 0.545

    # results dictionary
    averted_season = {}

    # array of treatment effectively:
    effectively_treated = np.array([treatment_prop[A[i]] for i in range(len(A))])

    # if our baseline policy is 2d treatment
    if policy_2d:
        effectively_treated = effectively_treated / treated_within_2d_prop

    # set age age group to 100% coverage:
    effectively_treated[age_idx] = 1

    # set new policy:
    new_policy = create_new_policy_matrices(np.zeros(len(A)), effectively_treated * prop_2d ,
                                            effectively_treated * (1-prop_2d))

    # run over all season:
    for season in seasons:
        # obtain season's parameters
        beta = fitted_parameters[str(season)]['beta']
        phi = fitted_parameters[str(season)]['phi']

        model_results = run_model(beta, phi * 10 ** (4), 1e-4, 1e-4,
                                  population_size=population_size[state],
                                  vaccination_coverage= vacc_coverage, policy=new_policy)

        new_I = np.array(model_results['new_I']).sum(axis=0)

        # calculating total symptomatic cases and per age group
        cases = np.array([new_I[[i, i + 5]].sum() for i in A.keys()])
        hospital = (new_I[5:] * hospitalizations['Low']) + \
                   (new_I[:5] * (1 - effectively_treated) * hospitalizations['High']) + \
                   (new_I[:5] * effectively_treated * hospitalizations['High_treated'])

        # insert baseline into the dictionary result:
        averted_season[str(season)] = {'Cases': baseline[str(season)]['Cases'] - cases,
                                       "Hospital": baseline[str(season)]['Hospital'] - hospital}

    return averted_season


def get_2d_baseline_results_age_group(state, seasons, fitted_parameters, vacc_coverage, all_treated = True):

    treated_within_2d_prop = 0.545

    # results dictionary
    baseline_season = {}

    # array of treatment effectively:
    effectively_treated = np.array([treatment_prop[A[i]] for i in range(len(A))])

    if all_treated:
        total_treated = effectively_treated / treated_within_2d_prop

    else:
        total_treated = effectively_treated

    new_policy = create_new_policy_matrices(np.zeros(len(A)), total_treated * 1, total_treated * 0)

    # run over all season:
    for season in seasons:

        # obtain season's parameters
        beta = fitted_parameters[str(season)]['beta']
        phi = fitted_parameters[str(season)]['phi']

        model_results = run_model(beta, phi * 10 ** (4), 1e-4, 1e-4,
                                  population_size=population_size[state],
                                  vaccination_coverage=vacc_coverage, policy= new_policy)

        new_I = np.array(model_results['new_I']).sum(axis = 0)

        # calculating total symptomatic cases and per age group
        cases = np.array([new_I[[i, i+5]].sum() for i in A.keys()])
        hospital = (new_I[5:] * hospitalizations['Low']) + \
                   (new_I[:5] * (1 - total_treated) * hospitalizations['High']) + \
                   (new_I[:5] * total_treated * hospitalizations['High_treated'])

        # insert baseline into the dictionary result:
        baseline_season[str(season)] = {'Cases': cases, "Hospital": hospital}

    return baseline_season

def weeks_for_year(year):
    """The function gets the year and returns the number of weeks in that year"""
    last_week = datetime.date(year, 12, 28)
    return last_week.isocalendar()[1]


#