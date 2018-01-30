import csv
import random
from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier

file_handle = open('data', 'r')
csv_file = csv.reader(file_handle, delimiter=',')
names = next(csv_file)
raw_data = [row for row in csv_file]
file_handle.close()
all_data = [[float(num) for num in row[0:len(row) - 1]] for row in raw_data]
all_targets = [(1 if row[len(row) - 1].strip() == "Normal" else 2) for row in raw_data]

predict_data = []
predict_targets = []
learn_data = []
learn_targets = []
n_predict_targets = 1000
n_learn_targets = 10000

for i in range(0, n_predict_targets):
    r = random.randrange(0, int(8528 - len(predict_data)/2)) if random.randrange(0, 2) == 0 else random.randrange(7000, len(all_data))
    predict_data.append(all_data.pop(r))
    predict_targets.append(all_targets.pop(r))

for i in range(0, n_learn_targets):
    r = random.randrange(0, int(8528 - len(predict_data)/2 - len(learn_data)/2)) if random.randrange(0, 2) == 0 else random.randrange(7000, len(all_data))
    learn_data.append(all_data.pop(r))
    learn_targets.append(all_targets.pop(r))

print(predict_targets)

clf = MLPClassifier()
clf.fit(learn_data, learn_targets)

result = clf.predict(predict_data)

successes = 0
for i in range(0, n_predict_targets):
    if result[i] == predict_targets[i]:
        successes = successes + 1

successes = successes / n_predict_targets
print(successes)
