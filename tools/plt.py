import csv

import matplotlib.pyplot as plt

def main(path):
    # Initialize lists to hold the data
    lengths = []
    accuracies = []

    # Read data from the log file
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            lengths.append(int(row[0]))
            accuracies.append(float(row[1]))

    # Plot the data
    plt.plot(lengths, accuracies, marker='o')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Accuracy = 0.5')  # Highlight the line at acc=0.5
    plt.xlabel('Length')
    plt.ylabel('Accuracy')
    plt.title('Length and Accuracy')
    plt.ylim(0, 1)  # Set y-axis range from 0 to 1
    plt.grid(True)
    plt.legend()  # Add legend to the plot
    plt.show()
    # Save the plot to the logs folder with the same name as the csv file
    plt.savefig('logs/log-0.8-15.png')

if __name__ == '__main__':
    path = 'logs/log-0.8-15.csv'
    main(path)