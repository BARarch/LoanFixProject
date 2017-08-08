''' LoanFix.py
    ====================================================================================================
    This is a python component of a much larger backend system used to retrieve records in a database of loans.  Key figures and trends are discovered in the data with pandas dataframes.
    
    ====================================================================================================

'''
import numpy as np
import pandas as pd
import datetime as dt
from dateutil import relativedelta as rd
from sklearn.linear_model import LinearRegression

df14 = pd.read_csv('data/LoanStats3c.csv', header=1, skipfooter=2, keep_default_na=False, na_values='.')
df15 = pd.read_csv('data/LoanStats3d.csv', header=1, skipfooter=2, keep_default_na=False, na_values='.')

df = df15.append(df14, ignore_index=True)



# Median Amount  of all Loans in DataSet: Simple Calcuation of median loan amount issued.
amounts = [x for x in df.loan_amnt]
np.median(amounts)
# 13750.0



# Fraction of Common Purpose Loan:  This script find the fraction of the most common loan use, which what empirically determined to be debt consolidation
purposes = df.purpose.unique()
purposeCount = {x:len(df[df.purpose == x]) for x in purposes}
purposeCount['debt_consolidation']/len(df)
# 0.5984644995462325



# Ratio of minimum average rate to the maximum average rate
purposeMeanInterest=[np.mean([float(str(x)[:-1]) for x in df[df.purpose == y].int_rate]) for y in purposes]
np.min(purposeMeanInterest)/np.max(purposeMeanInterest)
# 0.63997977670622108


# Difference in the fraction of the loans with a 36-month term between 2014 and 2015
len(df15[df15.term == " 36 months"])/len(df15) - len(df14[df14.term == " 36 months"])/len(df14)
#-0.017472334236841358



# Standard deviation of ratio of time spent in payment for all the loans in default:  All loans in default have to be collected and datetime is then used to determine the number months between the last payment and the term of the loans.
defaults = df[(df.loan_status != "Current") & (df.loan_status != "Fully Paid") & (df.loan_status != "In Grace Period")]

def monthsCount(rds):
    return rds.years * 12 + rds.months

def monthsBetween(issued, payment):
    if not payment:
        return 0
    return monthsCount(rd.relativedelta(dt.datetime.strptime(payment, '%b-%Y'), dt.datetime.strptime(issued, '%b-%Y')))

def termToMonths(term):
    return int(term[1:3])

termInd = list(defaults.dtypes.index).index('term') + 1
lastPaymentInd = list(defaults.dtypes.index).index('last_pymnt_d') + 1
issuedInd = list(defaults.dtypes.index).index('issue_d') + 1

# Compute list of payment/term ratios for all loans in default
paymntRatio = [monthsBetween(default[issuedInd], default[lastPaymentInd])/termToMonths(default[termInd]) for default in defaults.itertuples()]

# Compute standard deviation of list
np.std(paymntRatio)
# 0.19781779868425958



# Return on to-term Loans: This is the correlation factor between rate of return and interest rate for the loans.  This value is computed for all loans that are paid to term
toTerms = df[(df.loan_status == "Fully Paid")]
toTerms['returnRates'] = toTerms['total_pymnt'] / toTerms['loan_amnt']
toTerms['intRateFloat'] = pd.Series([float(val[:-1]) for val in toTerms['int_rate']], index=toTerms.index)

toTerms['returnRates'].corr(toTerms['intRateFloat'])
#0.54726352361989783



# Most Surprising Loan Purpose in a State: For each loan purpose we compute the probability that loan is isssued nationally and in each state.  By definition the state with the most suprizing loan purpose is the combination with the higest ratio between the state and national probabilities.  We determine vacations in the state of Hawaii are the most suprising because the probalilty of this purpose here is more than three times larger national figure.

def maxRatio(probs, counts, purposes, states):
    maxRat = 0
    stateMax = ''
    purposeMax = ''
    for state in states:
        for purpose in purposes:
            rat = probs[state][purpose] / probs['national'][purpose]
            if rat > maxRat and counts[state][purpose] >= 10:
                maxRat = rat
                stateMax = state
                purposeMax = purpose
    return maxRat, stateMax, purposeMax

purposes = df.purpose.unique()
states = df.addr_state.unique()

dataPurpose = {"loans":[len(df[df.purpose == x]) for x in purposes]}
purposeDf = pd.DataFrame(dataPurpose, index = purposes)
loanRecs = len(df)
purposeDf['national_prob'] = pd.Series([loans/loanRecs for loans in purposeDf['loans']], index=purposeDf.index)

dataStatePurpose = {state:[len(df[(df.purpose == purpose) & (df.addr_state == state)]) for purpose in purposes] for state in states}
purposeDfStates = pd.DataFrame(dataStatePurpose, index = purposes)

stateTotals= purposeDfStates.sum(axis=0)

statePurposeProbData = {state:list(purposeDfStates[state]/stateTotals[state]) for state in states}
statePurposeProbDf = pd.DataFrame(statePurposeProbData, index = purposes)
statePurposeProbDf['national'] = purposeDf['national_prob']

maxRatio(statePurposeProbDf, purposeDfStates, purposes, states)
# (3.2727693086839853, 'HI', 'vacation')



# Linear Model for Defaults Grouped by Subgrade: A new dataframe is created and converted to a matrix for linear regeression fitting useing scikit-learn.  We want to fit a linear model for the default rates of all 16 loan subgrade groups  

subGrades = df['sub_grade'].unique()

subGradeData = {'loans':[len(df[df['sub_grade'] == subGrade]) for subGrade in subGrades],
                'averageIntRates':[np.mean([float(x[:-1]) for x in df[df['sub_grade']==subGrade]['int_rate']]) for subGrade in subGrades],
                'defaults':[len(defaults[defaults['sub_grade'] == subGrade]) for subGrade in subGrades]}
subGradeDf = pd.DataFrame(subGradeData, index=subGrades)
subGradeDf['defaultRate'] = subGradeDf['defaults'] / subGradeDf['loans']

npMatrix = np.matrix(subGradeDf)
X, Y = npMatrix[:,0], npMatrix[:,3]
mdl = LinearRegression().fit(X,Y)
m = mdl.coef_[0]
b = mdl.intercept_
print("formula: y = " + str(m) + "x + " + str(b))

subGradeDf['predictedDefault'] = m * subGradeDf['averageIntRates'] + b
subGradeDf['dev'] = subGradeDf['predictedDefault'] - subGradeDf['defaultRate']

subGradeDf.loc[subGradeDf['dev'].idxmin()], subGradeDf.loc[subGradeDf['dev'].idxmax()]
#averageIntRates      26.874293
#defaults            172.000000
#loans               403.000000
#defaultRate           0.426799
#predictedDefault      0.396312
#dev                  -0.030487
#Name: G5, dtype: float64


