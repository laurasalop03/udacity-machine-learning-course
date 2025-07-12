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

    for i in range(len(predictions)):
        error = (predictions[i] - net_worths[i])**2
        cleaned_data.append((ages[i], net_worths[i], error))

    # sort by error
    cleaned_data.sort(key=lambda x: x[2])

    # we take away the 10% with the most error
    limit = int(len(cleaned_data) * 0.9)
    return cleaned_data[:limit]
