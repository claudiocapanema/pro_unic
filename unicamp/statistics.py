import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_metrics(df, acc_distributed_column):

    acc_average = df[acc_distributed_column].mean()
    acc_std = df[acc_distributed_column].var()
    #loss_average = df[loss_column].mean()

    return pd.DataFrame({acc_distributed_column + ' average': [acc_average], acc_distributed_column + ' variance': [acc_std]
                            #, 'Loss average': [loss_average]
                        })

def metrics(df, solution_column, acc_distributed_column, round_column, directory_results):

    df_statistics = df.groupby([solution_column, round_column]).apply(lambda e: calculate_metrics(e, acc_distributed_column))
    print("ESTATISTICAS")
    columns = df_statistics.columns.tolist()
    print(df_statistics)
    print([solution_column, round_column, directory_results] + columns)
    df_statistics.to_csv(directory_results + "solution_round_statistics.csv", index=True)
    df_statistics = df_statistics.reset_index()[[solution_column, round_column] + columns]
    line(df_statistics, filename=directory_result + 'line_acc_round average.png', x='Round', y='Accuracy distributed average', z='Solution')
    line(df_statistics, filename=directory_result + 'line_acc_round variance.png', x='Round', y='Accuracy distributed variance',
         z='Solution')
    #line(df_acc, filename=directory_result + 'line_acc_round.png', x='Round', y='Loss distributed average', z='Solution')


def read_csv(filename):

    return pd.read_csv(filename+".csv")

def line(df, filename, x, y, z, style=None):

    df[x] = df[x].astype(int)
    #df = df[df['Solution'] != 'qfedavg']
    print("\n", df, "\n")

    plt.figure()
    sns.set(style='whitegrid')

    figure = sns.lineplot(x=x , y=y, hue=z, data=df, style=style)
    if filename.find("loss") != -1:
        print("aqui")
        figure.set(yscale='log')
    figure = figure.get_figure()

    figure.savefig(filename, dpi=400)

if __name__ == "__main__":

    directory_data = "output_data1/"
    directory_result = "output_results/"
    loss_filename_list = ["FedAvg_loss", "QFedAvg_loss", "FedAdagrad_loss", "FedYogi_loss", "FedAvgM_loss"]
    acc_filename_list = ["FedAvg_acc", "QFedAvg_acc", "FedAdagrad_acc", "FedYogi_acc", "FedAvgM_acc"]

    # Loss

    df_loss = pd.DataFrame({'Solution': [], 'Round': [], 'Loss distributed': [], 'Loss centralized': []})
    for filename in loss_filename_list:
        filename = directory_data + filename
        df_loss = pd.concat([df_loss, read_csv(filename)], ignore_index=True)

    line(df_loss, filename=directory_result + 'line_loss_round.png', x='Round', y='Loss distributed', z='Solution')

    # Acc

    df_acc = pd.DataFrame({'Solution': [], 'Round': [], 'Accuracy distributed': [], 'Accuracy centralized': []})
    for filename in acc_filename_list:
        filename = directory_data + filename
        df_acc = pd.concat([df_acc, read_csv(filename)], ignore_index=True)
    line(df_acc, filename=directory_result + 'line_acc_round_ci.png', x='Round', y='Accuracy distributed', z='Solution')

    metrics(df_acc, solution_column='Solution', acc_distributed_column='Accuracy distributed', round_column='Round', directory_results=directory_result)



