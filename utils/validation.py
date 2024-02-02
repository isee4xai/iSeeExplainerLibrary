def validate_params(params, params_format):
    
    CAST_MAP={
    "int":int,
    "float":float,
    "string":str,
    "array":list
    }

    for param_name,param_format in params_format.items():

        param=params.get(param_name,"")

        if not param:
            if param_format["required"]: #if required, abort execution
                raise "Required parameter '" + param_name +"' was not included in the request." 
            else:
                param=param_format["default"] #set to default value if parameter was not included in request 
        else:
            try: #to cast
                param=CAST_MAP[param_format["type"]](param)
            except Exception as e:
                print(e)
                print("Defaulting '" + param_name +"' to " +str(param_format["default"]))
                params[param_name]=param_format["default"]
                continue

            if param_format["range"]:
                if param_format["type"]=="int" or param_format["type"]=="float":
                    if not (param>=param_format["range"][0] and param<=param_format["range"][1]): #out of range
                        print("Param out of range. Defaulting '" + param_name +"' to " +str(param_format["default"]))
                        param=param_format["default"]
                elif param_format["type"]=="string":
                    if param not in param_format["range"]:
                        print("Param out of range. Defaulting '" + param_name +"' to " +str(param_format["default"]))
                        param=param_format["default"]
                elif param_format["type"]=="array":
                    flag=True
                    for elem in param:
                        flag=elem in param_format["range"]
                        if not flag:
                            print(str(elem) + " out of range. Defaulting '" + param_name +"' to " +str(param_format["default"]))
                            param=param_format["default"]
                            break

        params[param_name]=param

    return params

