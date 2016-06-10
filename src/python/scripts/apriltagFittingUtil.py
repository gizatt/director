from director import robotsystem
from director import objectmodel as om
from director.consoleapp import ConsoleApp
from director import robotstate
from director import planplayback
from director import lcmUtils
from director import drcargs
from director import roboturdf
from director import simpletimer
from director import affordancemanager
from director import filterUtils
from director.timercallback import TimerCallback
from director.fieldcontainer import FieldContainer
from director.visualization import PolyDataItem
from director import kinectlcm
from director.uuidutil import newUUID
from director import transformUtils
from PythonQt import QtCore, QtGui, QtUiTools
from director import vtkAll as vtk
from director.debugVis import DebugData
from director import segmentation
import drc as lcmdrc
import bot_core
import numpy as np
import math
import argparse
import functools

from director.utime import getUtime

import scipy.interpolate

def addWidgetsToDict(widgets, d):

    for widget in widgets:
        if widget.objectName:
            d[str(widget.objectName)] = widget
        addWidgetsToDict(widget.children(), d)

class WidgetDict(object):

    def __init__(self, widgets):
        addWidgetsToDict(widgets, self.__dict__)

class ApriltagItem(PolyDataItem):
    def __init__(self, name, view, size, container=None):
        PolyDataItem.__init__(self, name, vtk.vtkPolyData(), view)

        self.addProperty('uuid', newUUID(), attributes=om.PropertyAttributes(hidden=True))
        self.addProperty('Origin', [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], attributes=om.PropertyAttributes(hidden=True))
        self.addProperty('Dimensions', [size, size, 0.001], attributes=om.PropertyAttributes(decimals=4, singleStep=0.005, minimum=0.0, maximum=1))
        self.addProperty('Subdivisions', 0, attributes=om.PropertyAttributes(minimum=0, maximum=1000))
        self.properties.setPropertyIndex('Dimensions', 0)
        self.properties.setPropertyIndex('Subdivisions', 1)

        om.addToObjectModel(self, parentObj=container)

        self.updateGeometryFromProperties()


    def updateGeometryFromProperties(self):
        d = DebugData()
        d.addCube(self.getProperty('Dimensions'), (0,0,0), subdivisions=self.getProperty('Subdivisions'))
        self.setPolyData(d.getPolyData())

        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(filterUtils.computeCentroid(d.getPolyData()))
        segmentation.makeMovable(self, t)



    def getPose(self):
        childFrame = self.getChildFrame()
        t = childFrame.transform if childFrame else vtk.vtkTransform()
        return transformUtils.poseFromTransform(t)

    def _onPropertyChanged(self, propertySet, propertyName):
        PolyDataItem._onPropertyChanged(self, propertySet, propertyName)
        if propertyName == 'Origin':
            self.updateGeometryFromProperties()
        if propertyName in ('Dimensions', 'Subdivisions'):
            self.updateGeometryFromProperties()

    def setPolyData(self, polyData):
        if polyData.GetNumberOfPoints():
            originPose = self.getProperty('Origin')
            pos, quat = originPose[:3], originPose[3:]
            t = transformUtils.transformFromPose(pos, quat)
            polyData = filterUtils.transformPolyData(polyData, t.GetLinearInverse())
        PolyDataItem.setPolyData(self, polyData)

    def repositionFromDescription(self, desc):
        position, quat = desc['pose']
        t = transformUtils.transformFromPose(position, quat)
        self.getChildFrame().copyFrame(t)

    def loadDescription(self, desc):
        self.syncProperties(desc, copyMode)
        self.repositionFromDescription(desc)
        self._renderAllViews()

    def syncProperties(self, desc):
        for propertyName, propertyValue in desc.iteritems():
            if self.hasProperty(propertyName) and (self.getProperty(propertyName) != propertyValue):
                self.setProperty(propertyName, propertyValue)

    def onRemoveFromObjectModel(self):
        PolyDataItem.onRemoveFromObjectModel(self)



class ApriltagPanel(object):

    def __init__(self):

        self.widget = QtGui.QTabWidget()
        
        self.tags = []

        self.buildWidget()

    def buildWidget(self):
        self.container = om.addContainer("Apriltags")
        for i in range(3):
            apriltagWidget = QtGui.QWidget()
            gridLayout = QtGui.QGridLayout(apriltagWidget)
            #gridLayout.setColumnStretch(0, 1)
            
            label = QtGui.QLabel("TABTODO")
            numericLabel = QtGui.QLabel(str(i))
            column = gridLayout.columnCount()
            gridLayout.addWidget(label, 0, column)
            gridLayout.addWidget(numericLabel, 2, column)
            #gridLayout.setColumnStretch(gridLayout.columnCount(), 1)
            self.widget.addTab(apriltagWidget, str(i))

            self.tags.append(ApriltagItem(str(i), view, 0.08, container=self.container))


class JointTeleopPanel(object):

    def __init__(self, stateModels, jointControllers):

        self.widget = QtGui.QTabWidget()

        self.stateModels = stateModels
        self.jointControllers = jointControllers

        self.buildTabWidget(stateModels, jointControllers)

    def buildTabWidget(self, stateModels, jointControllers):

        self.spinBoxMap = {}
        self.labelMap = {}
        self.poseMap = {}
        self.robotStateMap = {}
        self.robotJointControllerMap = {}

        for i, stateModel in enumerate(stateModels):
            groupName = stateModel.getProperty('Name')
            joints = stateModel.model.getJointNames()

            self.poseMap[groupName] = np.zeros((len(joints), 1))
            self.robotStateMap[groupName] = stateModel
            self.robotJointControllerMap[groupName] = self.jointControllers[i]

            jointGroupWidget = QtGui.QWidget()
            gridLayout = QtGui.QGridLayout(jointGroupWidget)
            #gridLayout.setColumnStretch(0, 1)

            for jointName in joints:
                label = QtGui.QLabel(jointName)
                spinbox = QtGui.QDoubleSpinBox()
                spinbox.setMinimum(-1000.)
                spinbox.setMaximum(1000.)
                spinbox.setSingleStep(0.01)
                row = gridLayout.rowCount()
                gridLayout.addWidget(label, row, 0)
                gridLayout.addWidget(spinbox, row, 1)
                self.spinBoxMap[(groupName, jointName)] = spinbox
                self.labelMap[(groupName, jointName)] = label

            #gridLayout.setRowStretch(gridLayout.rowCount(), 1)
            #gridLayout.setColumnStretch(1, 5)

            scroll = QtGui.QScrollArea()
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            scroll.setWidget(jointGroupWidget)

            self.widget.addTab(scroll, groupName)

        self.widget.usesScrollButtons = False
        self.signalMapper = QtCore.QSignalMapper()

        for robotJointTuple, spinbox in self.spinBoxMap.iteritems():
            spinbox.connect('valueChanged(double)', functools.partial(self.spinBoxChanged, robotJointTuple)) # self.signalMapper, 'map()')
            #self.signalMapper.setMapping(spinbox, robotJointTuple)

        #self.signalMapper.connect('mapped(const QString&)', self.spinBoxChanged)

    def showPose(self, robotName):
        self.robotJointControllerMap[robotName].setPose('teleop_pose', self.poseMap[robotName])

    def toJointIndex(self, robotName, jointName):
        return self.robotStateMap[robotName].model.getJointNames().index(jointName)

    def getJointValue(self, robotName, jointIndex):
        return self.poseMap[robotName][jointIndex]

    def spinBoxChanged(self, robotJointTuple, newVal):
        robotName = robotJointTuple[0]
        jointName = robotJointTuple[1]
        spinbox = self.spinBoxMap[robotJointTuple]
        jointIndex = self.toJointIndex(robotName, jointName)
        jointValue = spinbox.value

        self.poseMap[robotName][jointIndex] = jointValue
        self.showPose(robotName)
        self.updateSpinboxes()

    def updateSpinboxes(self):
        for robotJointTuple, spinbox in self.spinBoxMap.iteritems():
            jointName = robotJointTuple[1]
            robotName = robotJointTuple[0]
            jointIndex = self.toJointIndex(robotName, jointName)
            jointValue = self.getJointValue(robotName, jointIndex)

            spinbox.blockSignals(True)
            spinbox.setValue(jointValue)
            spinbox.blockSignals(False)


class ApriltagFittingPanel(object):

    def __init__(self):

        self.app = ConsoleApp()
        global view
        view = self.app.createView()
        self.view = view

        self.config = drcargs.getDirectorConfig()
        
        # load all relevant state models from config
        stateModels = []
        jointControllers = []
        if 'fittingConfig' in self.config.keys():
            for entry in self.config['fittingConfig']:
                mstatemodel, mjointcontroller = roboturdf.loadRobotModel(entry, self.view, urdfFile='../../'+self.config['fittingConfig'][entry]['urdf'], visible=True, parent="estimation", jointNames = self.config['fittingConfig'][entry]['drakeJointNames'])
                mjointcontroller.setPose(self.config['fittingConfig'][entry]['update_channel'], mjointcontroller.getPose('q_zero'))
                mjointcontroller.addLCMUpdater(self.config['fittingConfig'][entry]['update_channel'])
                stateModels.append(mstatemodel)
                jointControllers.append(mjointcontroller)

        self.jointTeleopPanel = JointTeleopPanel(stateModels, jointControllers)
        self.apriltagPanel = ApriltagPanel()
        self.widget = QtGui.QWidget()

        gl = QtGui.QGridLayout(self.widget)
        gl.addWidget(self.app.showObjectModel(), 0, 0, 4, 1) # row, col, rowspan, colspan
        gl.addWidget(self.view, 0, 1, 4, 3)
        gl.addWidget(self.jointTeleopPanel.widget, 0, 4, 3, 1)
        gl.addWidget(self.apriltagPanel.widget, 3, 4, 1, 1)
        #gl.setRowStretch(0,1)
        gl.setColumnStretch(1,5)

        # kinect management
        kinectlcm.init(self.view)

    def resetJointTeleopSliders(self):
        self.jointTeleopPanel.resetPoseToRobotState()
  

def main():

    p = ApriltagFittingPanel()
    p.widget.show()
    p.widget.resize(1400, 1400*9/16.0)
    p.app.setupGlobals(globals())
    p.app.start()


if __name__ == '__main__':
    main()

