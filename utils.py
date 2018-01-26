def split_token(token):
    """
    Splits a string into an array that contains newline tokens

    token - string that contains '\\' character
    """

    assert "\\" in token
    arr = token.split("\\")
    final_arr = []
    for x in arr:
        if x == "newline":
            final_arr.append("\\newline")
        elif "newline" in x:
            appended = False
            for i in range(len(x)-len("newline")):
                if x[i:i+len("newline")] == "newline":
                    if appended:
                        print("Failure in word:", x)
                    if i > 0:
                        final_arr.append(x[:i])
                        final_arr.append("\\"+x[i:])
                    else:
                        final_arr.append("\\"+x[:len("newline")])
                        final_arr.append(x[len("newline"):])
                    appended = True
        elif x != "":
            final_arr.append(x)
    return final_arr
