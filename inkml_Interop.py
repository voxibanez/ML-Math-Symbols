################################################################
# segGenerator.py
#
# Program that reads in inkml ground-truthed files and generate
# right or wrong segmented symbols.
#
#
#
################################################################
import sys
from inkml import *
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# list of applicable target classes
targets = (['\\div', '\\pm', '[', ']', '\\log', '\\tan', '\\beta', '\\alpha', 
            '\\int', '\\pi', ',', '\\cos', '\\sum', '\\theta', '.', '\\times',
            '\\sin', '\\sqrt', '=', ')', '(', '+', '-'])

class function:
    fullName = ""
    symbols = []
class symbol:
    def __init__(self,name,pictureData):
        self.name = name
        self.pictureData = pictureData
    name = ""
    pictureData = []

def generateRightSeg(ink, segName,real_point_weight, calculated_point_weight, lineFilling = True):
    """generate all one inkml file per symbol. Return the number of generated files."""
    
    segFunction = function()
    for seg in ink.segments.values():
        lab = seg.label
        pictureData = []

        # filter based on labels
        if lab not in targets:
            continue

        if (lab == ","):
            lab = "COMMA"

        #print lab
        mainArray = []
        for s in seg.strId:
            temp = []
            #Split data into 2d array
            for a in ink.strokes[s].split(','):
                temp.append(a.strip().split(" "))

            #Convert to int
            #i'm sorry
            for a in temp:
                a[0] = int(str(a[0].replace('.', '')))
                a[1] = int(str(a[1].replace('.', '')))
            mainArray.append(temp)


        extraPoints = []
        for s in mainArray:
            #NOTE: These points might be coordinates to lines instead of just points. If so, we will need to fill in the path between every two points (only if we're not getting good results)
            #Here is the line filling algorithm, enable or disable as needed from the function call
            #NOTE: in the future we may want to weight these as half of the normal points because these are computer calculated (implemented)
            if lineFilling:
                for x in range(0,len(s)):
                    if x >= len(s) - 1:
                        break
                    x1 = s[x][0]
                    y1 = s[x][1]
                    x2 = s[x+1][0]
                    y2 = s[x+1][1]
                    dx = x2 - x1
                    dy = y2 - y1

                    #Draw vertical or horizontal line
                    if x1-x2 == 0:
                        for y in range(y1, y2):
                            extraPoints.append([x1, y])
                    elif y1-y2 == 0:
                        for x in range(x1, x2):
                            extraPoints.append([x, y1])
                    else:
                        #For anything with a slope
                        for x in range(x1, x2):
                            y = int(y1 + dy * (x - x1) / dx)
                            extraPoints.append([x,y])

        mainArray2 = []
        for s in mainArray:
            mainArray2.extend(s)  # Normalize data (find min x and y and offset to 0)

        #Normalize array
        xmin = sys.maxint
        ymax = -sys.maxint - 1
        for a in mainArray2:
            if a[0] < xmin:
                xmin = a[0]
            if a[1] > ymax:
                ymax = a[1]
        for a in mainArray2:
            a[0] = a[0] - xmin
            a[1] = ymax - a[1]
        for a in extraPoints:
            a[0] = a[0] - xmin
            a[1] = ymax - a[1]

        #DEBUG plots of the array of coordinates before its converted into an array of pixels
        #plt.cla()
        #colors = itertools.cycle(["r", "b", "g"])
        #plt.title("Symbol: " + lab)
        #plt.scatter(*zip(*extraPoints), color=next(colors))
        #plt.scatter(*zip(*mainArray2))


        #Change from coordinates of black to grid of pixel on or off
        xmax = -sys.maxint - 1
        ymax = -sys.maxint - 1
        for a in mainArray2:
            if a[0] > xmax:
                xmax = a[0]
            if a[1] > ymax:
                ymax = a[1]
        newArray = [[0 for x in range(xmax + 1)] for y in range(ymax + 1)]
        #Give the computer generated points a half weight becuase it was calculated
        #This goes first so the main points can overwrite if there is overlap
        for a in extraPoints:
            newArray[len(newArray) - 1 - a[1]][a[0]] = calculated_point_weight
        for a in mainArray2:
            newArray[len(newArray) - 1 - a[1]][a[0]] = real_point_weight

        pictureData += [newArray]

        segFunction.symbols.append(symbol(lab,pictureData))
    segFunction.fullName = ink.truth
    return segFunction

def parseItem(item, real, calculated, fillLine):
    nb = 0

    try:
        f = Inkml(item.strip())
        nb = generateRightSeg(f, item, real, calculated, fillLine)

    except IOError:
        print
        "Can not open " + item.strip()
    except ET.ParseError:
        print
        "Inkml Parse error " + item.strip()

    return nb
