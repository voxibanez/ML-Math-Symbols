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

class function:
    fullName = ""
    symbols = []
class symbol:
    def __init__(self,name,pictureData):
        self.name = name
        self.pictureData = pictureData
    name = ""
    pictureData = []

def generateRightSeg(ink, segName, lineFilling = True):
    """generate all one inkml file per symbol. Return the number of generated files."""

    segFunction = function()
    for seg in ink.segments.values():
        lab = seg.label
        pictureData = []
        if (lab == ","):
            lab = "COMMA"
        for s in seg.strId:
            temp = []
            #Split data into 2d array
            for a in ink.strokes[s].split(','):
                temp.append(a.strip().split(" "))

            #Convert to int
            for a in temp:
                a[0] = int(a[0])
                a[1] = int(a[1])

            #Normalize data (find min x and y and offset to 0)
            xmin = sys.maxint
            ymin = sys.maxint
            for a in temp:
                if a[0] < xmin:
                    xmin = a[0]
                if a[1] < ymin:
                    ymin = a[1]
            xmin = xmin
            ymin = ymin
            for a in temp:
                a[0] = a[0] - xmin
                a[1] = a[1] - ymin

            #NOTE: These points might be coordinates to lines instead of just points. If so, we will need to fill in the path between every two points (only if we're not getting good results)
            #Here is the line filling algorithm, enable or disable as needed from the function call
            if lineFilling:
                for x in range(0,len(temp),2):
                    if x >= len(temp) - 1:
                        break
                    x1 = temp[x][0]
                    y1 = temp[x][1]
                    x2 = temp[x+1][0]
                    y2 = temp[x+1][1]
                    dx = x2 - x1
                    dy = y2 - y1
                    for x in range(x1, x2):
                        y = y1 + dy * (x - x1) / dx
                        temp.append([x,y])


            #Change from coordinates of black to grid of pixel on or off
            xmax = -sys.maxint - 1
            ymax = -sys.maxint - 1
            for a in temp:
                if a[0] > xmax:
                    xmax = a[0]
                if a[1] > ymax:
                    ymax = a[1]
            newArray = [[0 for x in range(xmax + 1)] for y in range(ymax + 1)]
            for a in temp:
                newArray[a[1]][a[0]] = 1
            pictureData += [newArray]
        segFunction.symbols.append(symbol(lab,pictureData))
    segFunction.fullName = ink.truth
    return segFunction

def parseItem(item, fillLine):
    n = -1
    nb = 0

    try:
        f = Inkml(item.strip())
        nb = generateRightSeg(f, item, fillLine)

    except IOError:
        print
        "Can not open " + item.strip()
    except ET.ParseError:
        print
        "Inkml Parse error " + item.strip()

    return nb
