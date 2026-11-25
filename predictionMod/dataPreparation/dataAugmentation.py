from .constants import *

def removeShortVids(dataLines):
    """
    this function removes videos which are too short

    get seperate lines
    they should be precutted to the last '2'
    they should be precutted to be withoug the name
    data and labels should not be separated
    """
    final_lines = []

    for data in dataLines:
        data = data.split(",")

        if (len(data) - DATA_PER_LABEL) > MIN_FRAMES_PER_VIDS:
            final_lines.append(data)

    return final_lines