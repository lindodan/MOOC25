#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    residual_errors = []
    for i in range(len(predictions)):
        residual_errors.append((abs(predictions[i] - net_worths[i]),i))

    residual_errors.sort()
    for keeping in range(int((len(net_worths)*0.9))):
        i = residual_errors[keeping][1]
        error = residual_errors[keeping][0][0]
        age = ages[i]
        net_worth = net_worths[i]
        cleaned_data.append((age,net_worth,error))


    
    return cleaned_data

