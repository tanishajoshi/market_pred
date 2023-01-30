import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd
#Extract data from s&p 500 index, clean and add columns
sp500 = yf.Ticker("^GSPC")

#queries all day from beginning when index was created
sp500 = sp500.history(period = "max") 

#plot = sp500.plot.line(y ="Close",use_index = True)

del sp500["Dividends"]
del sp500["Stock Splits"]


#target = will the stock go up or down tmrw
#create col Tmrw, with all data from 
sp500["Tomorrow"] = sp500["Close"].shift(-1)
print(sp500)
sp500["Target"] = (sp500["Tomorrow"]> sp500["Close"]).astype(int)
#only take rows in which the index (date) is 1990 jan 1st, 
# due to s&p shifts, we dont need data prior to this
sp500 = sp500.loc["1990-01-01":].copy()

#trains a bunch of individual decision trees 
# with randomized parameters and avg's the 
# results from those decision trees
# they run relatively quickly, and pick up 
# non linear tendencies in data

#initialize model
#n est = num of indiv decision trees we want to train, 
# higher the number higher the accuracy (sort of)
# min sample split = protects against overfitting
# random_state = if u run same model 2x, the random num generated will be in a predictable sequence
model = RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)

train = sp500.iloc[:-100] #all rows except last 100 in training set
test = sp500.iloc[-100:] #last 100 rows

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

RandomForestClassifier(min_samples_split=100, random_state=1)
#precision score: what % of the time were we accurate in prediction (market went up or down)
preds = model.predict(test[predictors])

preds = pd.Series(preds,index = test.index)
print(precision_score(test["Target"], preds))
combined = pd.concat([test["Target"], preds], axis=1)

combined.plot()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return combined

#train first model w 10 yrs of data
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0],step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500,model, predictors)
predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])
predictions["Target"].value_counts() /predictions.shape[0]

horizons = [2,5,60,250,1001]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_col=f"Close_Ratio_{horizon}"
    sp500[ratio_col] = sp500["Close"] / rolling_averages["Close"]

    trend_col = f"Trend_{horizon}"
    sp500[trend_col] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_col, trend_col]

sp500 = sp500.dropna()

new_model = RandomForestClassifier(n_estimators =200,min_samples_split=50,random_state = 1)

def improved_predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index,name="Predictions")
    combined = pd.concat([test["Target"],preds], axis=1)
    return combined

newpredictions = backtest(sp500,new_model,new_predictors)
print(newpredictions["Predictions"].value_counts())

precision_score(newpredictions["Target"],newpredictions["Predictions"])

