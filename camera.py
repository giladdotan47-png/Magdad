from math import sin as sin
from math import cos as cos
from math import pi as pi
import numpy as np

class Camera:
    _num = 0
    _xLoc = 0.0
    _yLoc = 0.0
    _zLoc = 0.0
    _compassOr = 0.0
    _heightOr = 0.0
    _horizontalAper = 0.0
    _vertApper = 0.0
    _vid=None
    _slope = None
    _xCamBase = None
    _yCamBase = None

    def __init__(self, num=None, xLoc=None, yLoc=None, zLoc=None, compassOr=None, heightOr=None, horizontalAper=None, vertApper=None, vid=None):
        if num is not None:
            self._num = num
        if xLoc is not None:
            self._xLoc = xLoc
        if yLoc is not None:
            self._yLoc = yLoc
        if zLoc is not None:
            self._zLoc = zLoc
        if compassOr is not None:
            self._compassOr = compassOr
        if heightOr is not None:
            self._heightOr = heightOr
        if horizontalAper is not None:
            self._horizontalAper = horizontalAper
        if vertApper is not None:
            self._vertApper = vertApper
        if vid is not None:
            self._vid = vid

        phi = self.degToRad(self._heightOr)
        theta = self.degToRad(self._compassOr)

        self._slope = np.array([sin(phi) * cos(theta),
                                sin(phi) * sin(theta),
                                cos(phi)])

        self._xCamBase = np.array([sin(phi) * cos(theta - (pi / 2)), sin(phi) * sin(theta - (pi / 2)), cos(phi)])
        self._yCamBase = np.array([sin(phi - (pi / 2)) * cos(theta), sin(phi - (pi / 2)) * sin(theta), cos(phi - (pi / 2))])

    def getX(self):
        return self._xLoc

    def getY(self):
        return self._yLoc

    def getZ(self):
        return self._zLoc

    def getCompOr(self):
        return self._compassOr

    def getHeightOr(self):
        return self._heightOr

    def getFrameNo(self, n):
        return self._vid[n]

    @staticmethod
    def degToRad(c):
        return (c * pi) / 180.0

    def getCamSlope(self):
        return self._slope

    def getXCamBase(self):
        return self._xCamBase

    def getYCamBase(self):
        return self._yCamBase