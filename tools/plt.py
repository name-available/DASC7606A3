import csv

import matplotlib.pyplot as plt

def plt_temperture(root_path):
    paths = {
        'temperature=0.8': root_path + '/log-0.8-10.csv',
        'temperature=1.2': root_path + '/log-1.2-10.csv',
        'temperature=1.6': root_path + '/log-1.6-10.csv'
    }

    for label, path in paths.items():
        x = []
        y = []
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x.append(int(row['len']))
                y.append(float(row['accuracy']))
            print(x, y)
        plt.plot(x, y, label=label)

    plt.xlabel('len')
    plt.ylabel('accuracy')
    plt.xlim(0, 300)
    plt.legend()
    plt.title('Accuracy with num_forward_passes = 10')
    plt.savefig(root_path + '/plt_temperature.png')
    plt.show()

def plt_passes(root_path):
    paths = {
        'num_forward_passes=10': root_path + '/log-0.8-10.csv',
        'num_forward_passes=12': root_path + '/log-0.8-12.csv',
        'num_forward_passes=15': root_path + '/log-0.8-15.csv',
        'num_forward_passes=20': root_path + '/log-0.8-20.csv'
    }

    for label, path in paths.items():
        x = []
        y = []
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x.append(int(row['len']))
                y.append(float(row['accuracy']))
        plt.plot(x, y, label=label)

    plt.xlabel('len')
    plt.ylabel('accuracy')
    plt.xlim(0, 300)
    plt.legend()
    plt.title('Accuracy with temperature = 0.8')
    plt.savefig(root_path + '/plt_passes.png')
    plt.show()

def plt_8_15(root_path):
    paths = {'num_forward_passes=15': root_path + '/log-0.8-15.csv'}

    for label, path in paths.items():
        x = []
        y = []
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x.append(int(row['len']))
                y.append(float(row['accuracy']))
        plt.plot(x, y, label = "Experiment")

    plt.axhline(y=0.5, color='r', linestyle='--', label='Accuracy=0.5')  # Highlight acc=0.5 line
    plt.xlabel('len')
    plt.ylabel('accuracy')
    plt.xlim(0, 1150)
    plt.legend()
    plt.title('Accuracy with temperature = 0.8, num_forward_passes = 15')
    plt.savefig(root_path + '/plt_0.8_15.png')
    plt.show()

def plt_12_10(root_path):
    paths = {'num_forward_passes=10': root_path + '/log-1.2-10.csv'}

    for label, path in paths.items():
        x = []
        y = []
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x.append(int(row['len']))
                y.append(float(row['accuracy']))
        plt.plot(x, y, label = "Experiment")

    plt.axhline(y=0.5, color='r', linestyle='--', label='Accuracy=0.5')  # Highlight acc=0.5 line
    plt.xlabel('len')
    plt.ylabel('accuracy')
    plt.xlim(0, 1150)
    plt.legend()
    plt.title('Accuracy with temperature = 1.2, num_forward_passes = 10')
    plt.savefig(root_path + '/plt_1.2_10.png')
    plt.show()


if __name__ == '__main__':
    path = 'logs'
    plt_8_15(path)
    plt_12_10(path)

