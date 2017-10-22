'''
Created on Oct 15, 2017

@author: surya
'''

import time
import arff
from dataset import Dataset
from tree import Tree
from collections import Counter

def main():
    sample_data_filepath = '../weather_data.arff'
    sample_test_data_filepath = '../weather_data.arff'
    sample_subset = '../weather_data_subset.arff'
    
    training_data_filepath = '../training_subsetD.arff'
    test_data_filepath = '../testingD.arff'
    test_sample = '../weather_test.arff'
    
    sample_target_attr_name = 'play'
    target_attr_name = 'Class'
    
    confidence_values = [0, 0.5, 0.8, 0.95, 0.99]
    #confidence_values = [0.0, 0.5, 0.95]
    for confidence in confidence_values:
        print "Confidence value: %.2f\n" % confidence
        start = time.clock()
        print "Reading input data..."
        #dataset = Dataset(sample_subset, sample_target_attr_name)
        dataset = Dataset(training_data_filepath, target_attr_name)
        print "Successfully parsed training data in %.3fs." % (time.clock() - start)
        
        start = time.clock()
        dtree = Tree()
        print "Learning tree by fitting data...\n"
        dtree.learn(dataset, confidence)
        print "Tree building successful in %.3fs." % (time.clock() - start)
        
        print "\nPrinting tree..\n\n", dtree.tree.ToString(), "\n"
        
        print "Reading test data.."
        start = time.clock()
        #test_data = arff.load(open(test_sample, 'rb'))
        test_data = arff.load(open(test_data_filepath, 'rb'))
        print "Successfully parsed test data in %.3fs." % (time.clock() - start)
        
        test_instances = test_data.get("data")
        
        print "Classifying test instances..\n"
        start = time.clock()
        predicted_values, actual_values = dtree.classify(test_instances, dataset.target_attr)
        print "Completed classification in %.3fs." % (time.clock() - start)
        
        true_pos, false_pos, true_neg, false_neg = getConfusionMatrix(predicted_values, actual_values, dataset.target_attr.values[0])
        accuracy = (true_pos + true_neg)/(len(test_instances))
        precision = 0 if true_pos == 0 else true_pos/(true_pos + false_pos)
        recall = 0 if true_pos == 0 else true_pos/(true_pos + false_neg)

        print "True positives: %.0f False positives: %.0f" % (true_pos, false_pos)
        print "False negatives: %.0f True negatives: %.0f" % (false_neg, true_neg)
        print "\nAccuracy: %.6f, Precision: %.6f, Recall: %.6f" % (accuracy, precision, recall)
        
        print "Predicted labels: ", Counter(predicted_values).most_common()
        print "Actual labels: ", Counter(actual_values).most_common()

        print "\n", "==" * 50, "\n"
        
'''
Computes the confusion matrix from the predicted and actual class labels.

@param predicted_values: List of predicted class labels for the test data
@param actual_values: List of actual class labels for the test data 
@param true_val: True class value

@return Returns the tuple containing true positives, false positives, true negatives, false negatives and list of indices containing misclassified instances.
'''

def getConfusionMatrix(predicted_values, actual_values, true_val):
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    for idx in range(len(predicted_values)):
        if predicted_values[idx] == actual_values[idx]:
            if predicted_values[idx] == true_val:
                tp += 1.0
            else:
                tn += 1.0
        else:
            if predicted_values[idx] == true_val:
                fp += 1.0
            else:
                fn += 1.0
    
    return tp, fp, tn, fn

if __name__ == "__main__":
    main()
