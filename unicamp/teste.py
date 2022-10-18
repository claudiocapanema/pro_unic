import seaborn as sns
from matplotlib import pyplot as plt


if __name__ == "__main__":
    plt.figure()
    fmri = sns.load_dataset("fmri")[['timepoint', 'signal', 'event']]
    fmri = fmri[fmri['timepoint']<2]
    print(fmri)
    fig = sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event")
    fig = fig.get_figure()
    fig.savefig("teste.png")