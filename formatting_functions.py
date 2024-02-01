import decimal


def format_float(input:float)->float:

    d = decimal.Decimal(f"{input}")
    if (d.as_tuple().exponent * -1) > 2:
        input = round(input,2)
        return input
    return input



def convert_to_binary(input:float,threshold:float)->list[int]:
    return (input >= threshold).astype(int)