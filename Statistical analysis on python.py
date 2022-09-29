import numpy as np, pandas as pd
import statistics as st
import statsmodels.api as sm
# from yahoofinancials import YahooFinancials
import statsmodels.tsa.api as smt

from scipy import stats  
import scipy.stats as ss
import scipy.stats.mstats as st1
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import statsmodels.stats._adnorm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white

from statsmodels.stats.diagnostic import acorr_breusch_godfrey

import warnings
warnings.filterwarnings('ignore')



# Applying Chow test for Market vs Stock
def get_rss(y, x):
    '''
    Inputs:
    formula_model: Is the regression equation that you are running
    data: The DataFrame of the independent and dependent variable

    Outputs: 
    rss: the sum of residuals
    N: the observations of inputs
    K: total number of parameters
    '''
    x = sm.add_constant(x)
    results=sm.OLS(y, x).fit()
    rss= (results.resid**2).sum()
    N=results.nobs
    K=results.df_model
    return rss, N, K, results


def Chow_Test(df, y, x, special_date, level_of_sig=0.05):
    
    from scipy.stats import f
    date=special_date
    x1=df[x][:date]
    y1=df[y][:date]
    x2=df[x][date:]
    y2=df[y][date:]

    RSS_total, N_total, K_total, results=get_rss(df[y], df[x])
    RSS_1, N_1, K_1, results1=get_rss(y1, x1)
    RSS_2, N_2, K_2, results2=get_rss(y2, x2)
    num=(RSS_total-RSS_1-RSS_2)/K_total
    den=(RSS_1+RSS_2)/(N_1+N_2-2*K_total)

    p_val = f.sf(num/den, 2, N_1+N_2-2*K_total)
    

    df['Before_Special'] = np.where(df.index<special_date , 'Before', 'After')
    g = sns.lmplot(x=x, y=y, hue="Before_Special", truncate=True, height=5, markers=["o", "x"], data=df)
    # return df

    if p_val<level_of_sig:
        print('The P vale {:3.5f} is lower than the level of significance {}. Therefore, reject the null that the coefficients are the same in the two periods are equal'.format(p_val,level_of_sig))
    else:
        print('The P vale {:3.5f} is higher than the level of significance {}. Therefore, accept the null that the coefficients are the same in the two periods are equal'.format(p_val,level_of_sig))

    return num/den, p_val

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        ss.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 


def Naive_Forecast(df_in_sample, df_out_sample, column):
    a=df_in_sample.tail(1).values
    df_naive_method_forecast=pd.DataFrame(np.repeat(a, df_out_sample.shape[0], axis=0))
    df_naive_method_forecast.columns=df_out_sample.columns
    # df_naive_method_forecast.head(3)

    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_naive_method_forecast[column], label='Naive Forecast Data')
    plt.legend(loc='best')
    plt.title("Naive Forecast")
    plt.show()
    df_naive_method_forecast.index=df_out_sample.index

    return df_naive_method_forecast


def Average_Forecast(df_in_sample, df_out_sample, column):
    a=df_in_sample.mean().values
    a=a.repeat(df_out_sample.shape[0]).reshape(a.shape[0], df_out_sample.shape[0]).transpose()
    df_Average_Forecast=pd.DataFrame(a)
    df_Average_Forecast.columns=df_out_sample.columns
    df_Average_Forecast.index=df_out_sample.index

    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_Average_Forecast[column], label='Simple Average Forecast Data')
    plt.legend(loc='best')
    plt.title("Simple Average Forecast")
    plt.show()
    df_Average_Forecast.index=df_out_sample.index

    return df_Average_Forecast


def Moving_Average_Forecast(df_in_sample, df_out_sample, column, window_leng):
    a=df_in_sample.rolling(window_leng).mean().iloc[-1].values
    a=a.repeat(df_out_sample.shape[0]).reshape(a.shape[0], df_out_sample.shape[0]).transpose()
    df_Moving_Average_Forecast=pd.DataFrame(a)
    df_Moving_Average_Forecast.columns=df_out_sample.columns
    df_Moving_Average_Forecast.index=df_out_sample.index

    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_Moving_Average_Forecast[column], label='Moving Average Forecast Data')
    plt.legend(loc='best')
    plt.title("Simple Moving Average Forecast")
    plt.show()

    df_Moving_Average_Forecast.index=df_out_sample.index

    return df_Moving_Average_Forecast


def Simple_Exponential_Smoothing_Forecast(df_in_sample, df_out_sample, column, level):
    a=[]
    for col in df_in_sample.columns:
        fit2 =smt.SimpleExpSmoothing(np.asarray(df_in_sample[col])).fit(smoothing_level=level, optimized=False)
        a.append(fit2.forecast())
    a=np.array(a)
    a=a.repeat(df_out_sample.shape[0]).reshape(a.shape[0], df_out_sample.shape[0]).transpose()
    df_Simple_Exponential_Smoothing_Forecast=pd.DataFrame(a)
    df_Simple_Exponential_Smoothing_Forecast.columns=df_out_sample.columns
    df_Simple_Exponential_Smoothing_Forecast.index=df_out_sample.index
    plt.figure(figsize=(7,4))
    plt.plot(df_in_sample.index, df_in_sample[column], label='Training Data')
    plt.plot(df_out_sample.index, df_out_sample[column], label='Actual Data')
    plt.plot(df_out_sample.index, df_Simple_Exponential_Smoothing_Forecast[column], label='Simple Exponential Smoothing Forecast Data')
    plt.legend(loc='best')
    plt.title("Simple Exponential Smoothing Forecast")
    plt.show()
    df_Simple_Exponential_Smoothing_Forecast.index=df_out_sample.index

    return df_Simple_Exponential_Smoothing_Forecast


# Here we are conducting a one tail test by speecifing if the alternative is "two-sided", "larger", or "smaller"

# def ttest_OLS(res, numberofbeta, X, value=0, alternative='two-sided', level_of_sig = 0.05):
#     results=np.zeros([2])
#     # numberofbeta represent the coeffiecent you would like to test 0 standts for interecept
#     results[0]=res.tvalues[numberofbeta]
#     results[1]=res.pvalues[numberofbeta]
#     if isinstance(X, pd.DataFrame):
#         column=X.columns[numberofbeta]
#     else:
#         column=numberofbeta
#     if alternative == 'two-sided':
#         if results[1]<level_of_sig:
#             print("We reject the null hypothesis that the Selected Coefficient: {} is equal to {} with a {} % significance level".format(column, value, level_of_sig*100))
#         else: print("We accept the null hypothesis that the Selected Coefficient: {} is equal to {} with a {} % significance level".format(column, value, level_of_sig*100))
#     elif alternative == 'larger':
#         if (results[0] > 0) & (results[1]/2 < level_of_sig):
#             print("We reject the null hypothesis that the Selected Coefficient: {} is less than {} with a {} % significance level".format(column, value, level_of_sig*100))
#         else: print("We accept the null hypothesis that the Selected Coefficient: {} is less than {} with a {} % significance level".format(column, value, level_of_sig*100))

#     elif alternative == 'smaller':
#         if (results[0] < 0) & (results[1]/2 < level_of_sig):
#             print("We reject the null hypothesis that the Selected Coefficient: {} is more than {} with a {} % significance level".format(column, value, level_of_sig*100))
#         else: print("We accept the null hypothesis that the Selected Coefficient: {} is more than {} with a {} % significance level".format(column, value, level_of_sig*100))

def Simple_ttest_Ols(results, hypothesis, alternative='two-sided', level_of_sig = 0.05):
    results1=np.zeros([2])
    t_test = results.t_test(hypothesis)
    results1[0]=t_test.tvalue
    results1[1]=t_test.pvalue
    if alternative == 'two-sided':
        if results1[1]<level_of_sig:
            print("We reject the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
        else: print("We accept the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
    elif alternative == 'larger':
        if (results1[0] > 0) & (results1[1]/2 < level_of_sig):
            print("We reject the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
        else: print("We accept the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))

    elif alternative == 'smaller':
        if (results1[0] < 0) & (results1[1]/2 < level_of_sig):
            print("We reject the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))
        else: print("We accept the null hypothesis: {} with a {} % significance level".format(hypothesis, level_of_sig*100))

    pass



def Joining_columns(df1, x, y=None,Name_of_new_column=None):
    # Find all columns except x
    f=df1.columns.to_list()
    f.remove(x)
    if y!=None:
        df2=pd.melt(df1, id_vars=[x], value_vars=y)  
    else:
        df2=pd.melt(df1, id_vars=[x], value_vars=f)
    if Name_of_new_column!=None:
        df2=df2.rename(columns={"value": Name_of_new_column})
    return df2



def get_betas_SLR(df1, x, column=None):
    if column==None:
    # Choose all the columns except the x
        f=df1.columns.to_list()
        f.remove(x)
        column=f
    A=np.zeros([len(column)])
    j=0
    for i in column:
        formula ='Q("'+ i+'"'+') ~ Q("Excess Market Returns")'
        results = smf.ols(formula, df1).fit()
        A[j]=results.params[1]
        j=j+1
        
    A=pd.DataFrame(data=A,columns=['Beta'], index=column)
    return A


def Get_indicators(BB, indicat):
    # BB represents the BB=yahoo_financials.get_key_statistics_data()
    V=np.zeros([len(BB)])
    j=0
    for i in BB.keys():
        V[j]=BB[i][indicat]
        j=j+1
    return V

# Examples V=Get_indicators(BB, 'priceToBook')


# Get Data from a dictionary downloaded from yahoo finance
def Get_Dataframe_of_tickes(tickers):
    yahoo_financials = YahooFinancials(tickers)
    BB=yahoo_financials.get_key_statistics_data()
    dict_of_df = {k: pd.DataFrame.from_dict(v, orient='index') for k,v in BB.items()}
    df = pd.concat(dict_of_df, axis=1)
    return df

# Examples
# tickers = ['AAPL', 'WFC', 'F', 'FB', 'DELL', 'SNE','NOK', 'MSFT', 'JPM', 'GE', 'BAC']
# Name=['Apple', 'Wells_Fargo_Company', 'Ford Motor Company', 'Facebook', 'Dell Technologies', 'Sony', 'Nokia', 'Microsoft', 'JPMorgan Chase & Co', 'General Electric', 'Bank of America']
# df=RF.Get_Dataframe_of_tickes(tickers)


def Get_Yahoo_stats(tickers):
    yahoo_financials = YahooFinancials(tickers)
    f=['get_interest_expense()', 'get_operating_income()', 'get_total_operating_expense()', 'get_total_revenue()', 'get_cost_of_revenue()', 'get_income_before_tax()', 'get_income_tax_expense()', 'get_gross_profit()', 'get_net_income_from_continuing_ops()', 'get_research_and_development()', 'get_current_price()', 'get_current_change()', 'get_current_percent_change()', 'get_current_volume()', 'get_prev_close_price()', 'get_open_price()', 'get_ten_day_avg_daily_volume()', 'get_three_month_avg_daily_volume()', 'get_stock_exchange()', 'get_market_cap()', 'get_daily_low()', 'get_daily_high()', 'get_currency()', 'get_yearly_high()', 'get_yearly_low()', 'get_dividend_yield()', 'get_annual_avg_div_yield()', 'get_five_yr_avg_div_yield()', 'get_dividend_rate()', 'get_annual_avg_div_rate()', 'get_50day_moving_avg()', 'get_200day_moving_avg()', 'get_beta()', 'get_payout_ratio()', 'get_pe_ratio()', 'get_price_to_sales()', 'get_exdividend_date()', 'get_book_value()', 'get_ebit()', 'get_net_income()', 'get_earnings_per_share()', 'get_key_statistics_data()']
    i=0
    exec('d=yahoo_financials.'+f[i], locals(), globals())
    col=f[i].replace("get_","").replace("()","")
    A=pd.DataFrame.from_dict(d, orient='index', columns=[col])
    for i in range(1,3):
        exec('d=yahoo_financials.'+f[i], locals(), globals())
        col=f[i].replace("get_","").replace("()","").replace("_"," ")
        B=pd.DataFrame.from_dict(d, orient='index', columns=[col])
        A= pd.concat([A, B], axis=1, sort=False)
    return A    

# from yahoofinancials import YahooFinancials
# tickers = ['AAPL', 'WFC', 'F', 'FB', 'DELL', 'SNE']
# Get_Yahoo_stats(tickers)

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
            
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

# Examples

# data = sm.datasets.longley.load_pandas()
# df1=data.data
# formula = 'GNP ~ YEAR + UNEMP + POP + GNPDEFL'
# results = smf.ols(formula, df1).fit()
# print(results.summary())
# res = RF.forward_selected(df1, 'GNP')
# print(res.model.formula)
# print(res.rsquared_adj)
# print(res.summary())


def GQTest(lm2, level_of_sig=0.05, sp=None):
    name = ['F statistic', 'p-value']
    test = sms.het_goldfeldquandt(lm2.resid, lm2.model.exog, split=sp)
    R=lzip(name, test)

    if test[1]>level_of_sig:
        print('The P vale of this test is {:3.5f}, which is greater than the level of significance {} therefore, we accept the null that the error terms are homoscedastic'.format(test[1], level_of_sig))
    else:
        print('The P vale of this test is {:3.5f}, which is smaller than the level of significance {} therefore, we reject the null, hence the error terms are hetroscedastic'.format(test[1], level_of_sig))
    return R


def WhiteTest(statecrime_model, level_of_sig=0.05):
    white_test = het_white(statecrime_model.resid,  statecrime_model.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    R=dict(zip(labels, white_test))

    if white_test[3]>level_of_sig:
        print('The P vale of this test is {:3.5f}, which is greater than the level of significance {} therefore, we accept the null that the error terms are homoscedastic'.format(white_test[3], level_of_sig))
    else:
        print('The P vale of this test is {:3.5f}, which is smaller than the level of significance {} therefore, we reject the null, hence the error terms are hetroscedastic'.format(white_test[3], level_of_sig))
    return R


def Plot_resi_corr(results):
    res_min_1=results.resid[:-1]
    res_plus_1=results.resid[1:]
    data1=pd.DataFrame(np.column_stack((res_min_1.T,res_plus_1.T)), columns=['u_t-1','u_t'])
    sns.set()
    plt.figure(figsize=(5,5))
    ax = sns.scatterplot(x='u_t-1', y='u_t', data=data1)
    pass

def Plot_resi_corr_time(results,df):
    C=pd.DataFrame(results.resid, index=df.index, columns=['Residuals'])
    C.plot(figsize=(10,5), linewidth=1.5, fontsize=10)
    plt.xlabel('Date', fontsize=10);
    return C


def Breusch_Godfrey(results, level_of_sig=0.05, lags=None):
    A=acorr_breusch_godfrey(results, nlags=lags)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    R=dict(zip(labels, A))

    if A[3]>level_of_sig:
        print('The P vale of this test is {:3.5f}, which is greater than the level of significance {} therefore, we accept the null that the error terms are not Auto-corrolated'.format(A[3], level_of_sig))
    else:
        print('The P vale of this test is {:3.5f}, which is smaller than the level of significance {} therefore, we reject the null, hence the error terms are Auto-corrolated'.format(A[3], level_of_sig))
    return R

def Create_lags_of_variable(MainDF_first_period, lags, column):
    # Crete a new dataframe based on the lag variables
    x=column
    if type(lags) == int:
        j=lags
        values=MainDF_first_period[x]
        dataframe = pd.concat([values.shift(j), values], axis=1)
        dataframe.columns = [x+' at time t-'+str(j), x+' at time t']
        dataframe=dataframe.dropna()
    else:
        values=MainDF_first_period[x]
        dataframe=values
        for j in lags:
            dataframe = pd.concat([values.shift(j), dataframe], axis=1)
        c=[x+' for time t-'+str(j) for j in range(len(lags),-1,-1)]
        dataframe.columns=c
        dataframe=dataframe.dropna()
    return dataframe
