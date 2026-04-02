



def llama70b(rule_clauses, sensor_list):


    part1 = "Please write a python function to analyze a pandas data frame. " \
    "The function should identify the time period for which the following conditions are satisfied: \n"

    
    part2 = "\n\nIn the data frame that will be passed to the function, the available column names are:\n"

    sensor_list.append("timestamp_str")
    sensor_list.append("timestamp")

    prompt = [part1] + rule_clauses + [part2] + sensor_list

    return prompt 
