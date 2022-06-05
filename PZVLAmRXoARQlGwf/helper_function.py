import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()


def plot_predictions(y_train, y_val, y_test, y_train_predict, y_val_predict, y_test_predict):
    fig =plt.figure(figsize = (30,10))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
   
    # Plotting training data
    plt.subplot(1,3,1)
    ax = sns.lineplot(x =np.arange(0,len(y_train)), y=y_train, label="Date", color="royalblue")
    ax =sns.lineplot(x =np.arange(0, len(y_train)), y =y_train_predict.reshape(-1),
                     label="Training Prediction (LSTM)", color="tomato")
    ax.set_title("Stock price", size=14 , fontweight= "bold")
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Cost (USD $)", size=14)
    ax.set_xticklabels("", size=10)
    
    #ploting for Validation data
    plt.subplot(1,3,2)
    ax = sns.lineplot(x =np.arange(0,len(y_val)), y=y_val, label="Date", color="royalblue")
    ax =sns.lineplot(x =np.arange(0, len(y_val)), y =y_val_predict.reshape(-1),
                     label="Validation Data Prediction (LSTM)", color="tomato")
    ax.set_title("Stock price", size=14 , fontweight= "bold")
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Cost (USD $)", size=14)
    ax.set_xticklabels("", size=10)
    
    #ploting for Test data
    plt.subplot(1,3,3)
    ax = sns.lineplot(x =np.arange(0,len(y_test)), y=y_test, label="Date", color="royalblue")
    ax =sns.lineplot(x =np.arange(0, len(y_test)), y =y_test_predict.reshape(-1),
                     label="Test Data Prediction (LSTM)", color="tomato")
    ax.set_title("Stock price", size=14 , fontweight= "bold")
    ax.set_xlabel("Days", size=14)
    ax.set_ylabel("Cost (USD $)", size=14)
    ax.set_xticklabels("", size=10)
    
    plt.show()
    



def get_bollinger_bands(df, days):
    df['rolling_avg'] = df['predicted'].rolling(days).mean()
    df['rolling_std'] = df['predicted'].rolling(days).std()
    df['Upper Band']  = df['rolling_avg'] + (df['rolling_std'] * 2)
    df['Lower Band']  = df['rolling_avg'] - (df['rolling_std'] * 2)
    df = df.dropna()
    return df
    
    
    
    
def plot_bollinger(df):
    plt.style.use("fivethirtyeight")
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplots(111)
    
    # get inde value for the X axis for facebook dataframe
    x_axis = df.index.get_level_values(0)
    ax.fill_between(x_axis, df["Upper Band"], df["Lower Band"], color="grey")
    
    # Plot adjust closing price and moving averages
    ax.plot(x_axis, df["predicted"], color="blue", lw=2)
    ax.plot(x_axis , df["rolling_avg"], color="black", lw=2)
    
    # Set Title & show the  iMage
    ax.set_title("20 Days Bollinger Band")
    ax.set_xlabel("Date (Year/Month)")
    ax.set_ylable("Price (USD $)")
    
   
   
    
def implement_bb_strategy(data, lower_bb, upper_bb):
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0
    
    for i in range(1,len(data)):
        if data[i-1] > lower_bb[i-1] and data[i] < lower_bb[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        elif data[i-1] < upper_bb[i-1] and data[i] > upper_bb[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)
            
    return buy_price, sell_price, bb_signal




def plot_bb_strategy(df, buy_price, sell_price):
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (20, 10)
    df['predicted'].plot(label = 'PREDICTED PRICES', alpha = 0.3)
    df['Upper Band'].plot(label = 'UPPER BB', linestyle = '--', linewidth = 1, color = 'black')
    df['rolling_avg'].plot(label = 'MIDDLE BB', linestyle = '--', linewidth = 1.2, color = 'grey')
    df['Lower Band'].plot(label = 'LOWER BB', linestyle = '--', linewidth = 1, color = 'black')
    plt.scatter(df.index[1:], buy_price, marker = '^', color = 'green', label = 'BUY', s = 200)
    plt.scatter(df.index[1:], sell_price, marker = 'v', color = 'red', label = 'SELL', s = 200)
    plt.title('BB STRATEGY TRADING SIGNALS')
    plt.legend(loc = 'upper left')
    plt.show()




def capital_gain_predicted_prices(df1, df2):
    # Define the total capital gain as the final value minus the initial value divided by the initial value
    actual_returns = (df1.iloc[-1] - df1.iloc[0])/df1.iloc[0]
    # Predicted returns 
    predicted_returns = (df2.iloc[-1] - df2.iloc[0])/df2.iloc[0]
    # Show the value as percentage
    actual_returns = round(actual_returns*100,2)
    predicted_returns = round(predicted_returns*100, 2)
    print('The actual returns had a value of {}% and the predicted returns had a value {}%'.format(actual_returns, 
                                                                                                predicted_returns))


def get_bollinger_df(actuals, predictions, days):
    pred_df = pd.DataFrame(predictions)
    pred_df.columns = ['predicted']
    pred_df['actuals'] = actuals
    pred_df = get_bollinger_bands(pred_df, days)
    
    pred_df = pred_df.reset_index(drop=True)
    
#     display(pred_df.head())
    buy_price, sell_price, bb_signal = \
    implement_bb_strategy(pred_df['predicted'], pred_df['Lower Band'], pred_df['Upper Band'])
    
    plot_bb_strategy(pred_df, buy_price, sell_price )
    
    
    pred_df = pred_df.iloc[1:].reset_index(drop=True)
    pred_df['bb_signal'] = bb_signal
    
    
    capital_gain_predicted_prices(df1 = pred_df['actuals'], df2 = pred_df['predicted'])
    
    return pred_df