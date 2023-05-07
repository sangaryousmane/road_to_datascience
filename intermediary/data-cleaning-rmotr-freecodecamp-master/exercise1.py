import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_data_from_file(file_name) -> pd.DataFrame:
    df = pd.read_csv(
        file_name,
        index_col=0,
        parse_dates=True
    )
    return df;



def manipulate_data():
    df=load_data_from_file('data/btc-eth-prices-outliers.csv');
    df.plot(figsize=(16,9))
    plt.show()

def vivid_visualize_outlier():
    df=load_data_from_file('data/btc-eth-prices-outliers.csv');
    df.loc['2017-12': '2017-12-15'].plot(y='Ether', figsize=(16,9));
    plt.show()


def cleaning_data():
    df=load_data_from_file('data/btc-eth-prices-original.csv');
    cleaned_data = df.drop(pd.to_datetime(['2017-12-28', '2018-03-04']))
    # cleaned_data.plot(figsize=(16,9))
    cleaned_data.plot(y='Bitcoin', figsize=(16,9))
    plt.show()


def using_seaborn_for_plotting():
    df=load_data_from_file('data/btc-eth-prices-original.csv');
    fig, ax=plt.subplots(figsize=(15, 7));
    sns.histplot(df['Ether'], ax=ax);
    plt.show()


using_seaborn_for_plotting()