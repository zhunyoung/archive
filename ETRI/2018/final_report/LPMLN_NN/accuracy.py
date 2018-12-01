# python accuracy.py result_pro.txt label.txt

import sys

f_result = sys.argv[1]
f_label = sys.argv[2]

print("The result file with probabilities is: "+f_result)
print("The label file is: "+f_label)

idx2label = {}

with open(f_result) as fr:
	lines = fr.readlines()
	for line in lines:
		line = line.strip()
		features = line.replace('(',')').replace(') :  ',')').replace('\n','').split(")")
		print(features)

		predicate = features[0]
		idx, label = features[1].replace(', ',',').split(",")
		pro = features[2].strip()

		# print(predicate + "\n"+ idx+"\n"+label+"\n" + pro)
		if predicate in idx2label:
			if idx in idx2label[predicate]:
				# compare the new probability with the old one
				if float(pro) > idx2label[predicate][idx][1]:
					idx2label[predicate][idx] = [label, float(pro)]
			else:
				idx2label[predicate][idx] = [label, float(pro)]
		else:
			idx2label[predicate] = {}
			idx2label[predicate][idx] = [label, float(pro)]
	# print(idx2label)

with open(f_label) as fr:
	countCorrect = 0
	countFalse = 0
	countUnpredict = 0

	TP = 0
	TN = 0

	FP = 0
	FN = 0

	lines = fr.readlines()
	for line in lines:
		features = line.replace('(',')').replace('\n','').split(")")
		print(features)
		if len(features) <2:
			pass
			# print("!!!!line:\n")
			# print(line)
		else:
			predicate = features[0]
			idx, label = features[1].replace(', ',',').split(",")

			if predicate in idx2label:
				if idx in idx2label[predicate]:
					if label == idx2label[predicate][idx][0]:
						countCorrect += 1
						if label == "1":
							TP += 1
						else:
							TN += 1
							
					else:
						countFalse += 1
						if label == "0":
							FN += 1
						else:
							FP += 1
				else:
					countUnpredict += 1
					print("Index "+ idx + " not shown in the result file -- no prediction was made.")
			else:
				print("Predicate" + predicate + " not shown in the result file.")

	print("TP: " + str(TP))
	print("TN: " + str(TN))
	print("FP: " + str(FP))
	print("FN: " + str(FN))

	print("Correct Prediction: " + str(countCorrect))
	print("False Prediction: " + str(countFalse))
	print("Not Predicted: " + str(countUnpredict))
	print("Precision: " + str(TP/(TP+FP)))
	print("Recall: " + str(TP/(TP+FN)))

	print("Accuracy: "+ str(float(countCorrect)/(countCorrect+countFalse)))
