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
from director import cameraview
from bot_core import robot_state_t
import drc as lcmdrc
import bot_core
import numpy as np
import math
import argparse
import functools

from director.utime import getUtime

import scipy.interpolate

def getBotFrame(frameName):
    t = vtk.vtkTransform()
    t.PostMultiply()
    cameraview.imageManager.queue.getTransform(frameName, 'local', t)
    return t

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

    def __init__(self, name, stateModels):

        self.widget = QtGui.QWidget()
        self.name = name
        self.stateModels = stateModels

        self.buildWidget()

        self.timerCallback = TimerCallback(30)
        self.timerCallback.callback = self.update
        self.timerCallback.start()

    def buildWidget(self):
        self.container = om.addContainer("Apriltags")

        gridLayout = QtGui.QGridLayout(self.widget)
        #gridLayout.setColumnStretch(0, 1)
        
        robotlabel = QtGui.QLabel("Attached robot:")
        linklabel = QtGui.QLabel("Attached link:") 

        self.robotCombo = QtGui.QComboBox()
        self.linkCombo = QtGui.QComboBox()
        for stateModel in self.stateModels:
            robotName = stateModel.getProperty('Name')
            self.robotCombo.addItem(robotName)
        self.robotCombo.setCurrentIndex(0)
        self.populateLinkComboBox()
        self.robotCombo.connect('currentIndexChanged(int)', functools.partial(self.populateLinkComboBox))

        xyzlabel = QtGui.QLabel("XYZ: ")
        self.xyzLabel = QtGui.QLabel("")

        rpylabel = QtGui.QLabel("RPY: ")
        self.rpyLabel = QtGui.QLabel("")

        gridLayout.addWidget(robotlabel, 0, 0)
        gridLayout.addWidget(self.robotCombo, 0, 1)
        gridLayout.addWidget(linklabel, 1, 0)
        gridLayout.addWidget(self.linkCombo, 1, 1)
        gridLayout.addWidget(xyzlabel, 2, 0)
        gridLayout.addWidget(self.xyzLabel, 2, 1)
        gridLayout.addWidget(rpylabel, 3, 0)
        gridLayout.addWidget(self.rpyLabel, 3, 1)

        self.apriltag = ApriltagItem(self.name, view, 0.08, container=self.container)

    def populateLinkComboBox(self, extra=None):
        stateModel = self.stateModels[self.robotCombo.currentIndex]
        self.linkCombo.clear()
        for linkname in stateModel.model.getLinkNames():
            self.linkCombo.addItem(linkname)

    def update(self):
        linkFrame = transformUtils.copyFrame( self.stateModels[self.robotCombo.currentIndex].getLinkFrame(self.linkCombo.currentText ))
        apriltagFrame = transformUtils.copyFrame(self.apriltag.getChildFrame().transform)

        apriltagFrame.PostMultiply()
        apriltagFrame.Concatenate( linkFrame.GetLinearInverse() )

        xyz = transformUtils.getNumpyFromTransform(apriltagFrame)[0:3, 3]
        rpy = transformUtils.rollPitchYawFromTransform(apriltagFrame)
        self.xyzLabel.setText("%06.3f, %06.3f, %06.3f" % (xyz[0], xyz[1], xyz[2]))
        self.rpyLabel.setText("%06.3f, %06.3f, %06.3f" % (rpy[0]*180./math.pi, rpy[1]*180./math.pi, rpy[2]*180./math.pi))

class JointTeleopPanel(object):

    def __init__(self, stateModels, jointControllers):

        self.widget = QtGui.QTabWidget()

        self.stateModels = stateModels
        self.jointControllers = jointControllers

        # hacks to get this experiment up and running...
        self.estRobotStateRobotName = "arm"
        self.estRobotStateJointNames = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.buildTabWidget(stateModels, jointControllers)

        lcmUtils.addSubscriber('EST_ROBOT_STATE', robot_state_t, self.handleEstRobotState)

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
                spinbox.setMinimum(-10.)
                spinbox.setMaximum(10.)
                spinbox.setSingleStep(0.005)
                spinbox.setDecimals(5)
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

    def handleEstRobotState(self, msg):
        for jointname in self.estRobotStateJointNames:
            for i in range(msg.num_joints):
                if msg.joint_name[i] == jointname:
                    self.poseMap[self.estRobotStateRobotName][self.toJointIndex(self.estRobotStateRobotName, jointname)] = msg.joint_position[i]
                    break
        self.showPose(self.estRobotStateRobotName)
        self.updateSpinboxes()

class ApriltagFittingPanel(object):

    def __init__(self):

        self.app = ConsoleApp()
        global view
        view = self.app.createView()
        self.view = view

        #cameraview.init()

        self.config = drcargs.getDirectorConfig()
        
        # load all relevant state models from config
        stateModels = []
        jointControllers = []
        if 'fittingConfig' in self.config.keys():
            for entry in self.config['fittingConfig']:
                mstatemodel, mjointcontroller = roboturdf.loadRobotModel(entry, self.view, urdfFile='../../'+self.config['fittingConfig'][entry]['urdf'], visible=True, parent="estimation", jointNames = self.config['fittingConfig'][entry]['drakeJointNames'])
                mjointcontroller.setPose(self.config['fittingConfig'][entry]['update_channel'], mjointcontroller.getPose('q_zero'))
                #mjointcontroller.addLCMUpdater(self.config['fittingConfig'][entry]['update_channel'])
                stateModels.append(mstatemodel)
                jointControllers.append(mjointcontroller)

        self.jointTeleopPanel = JointTeleopPanel(stateModels, jointControllers)
        self.apriltagPanel = ApriltagPanel("123", stateModels)
        self.widget = QtGui.QWidget()

        gl = QtGui.QGridLayout(self.widget)
        gl.addWidget(self.app.showObjectModel(), 0, 0, 4, 1) # row, col, rowspan, colspan
        gl.addWidget(self.view, 0, 1, 4, 3)
        sa = QtGui.QScrollArea()
        sa.setWidget(self.jointTeleopPanel.widget)
        gl.addWidget(sa, 0, 4, 3, 1)
        gl.addWidget(self.apriltagPanel.widget, 3, 4, 1, 1)
        #gl.setRowStretch(0,1)
        gl.setColumnStretch(1,5)

        # kinect management
        kinectlcm.init(self.view)

        #tr = getBotFrame('robot_yplus_tag')
        #tagTranslation = list(transformUtils.poseFromTransform(tr)[0])
        #tagRpy = [k*(180/math.pi) for k in list(transformUtils.rollPitchYawFromTransform(tr))] #[-90,0,90]
        tagToWorld = transformUtils.frameFromPositionAndRPY([0, 0, 0], [-90, 0, 90])#tagTranslation, tagRpy)

        cameraToTag = transformUtils.transformFromPose([0.08, -0.04, 1.38], [0.89, 0.2, -0.375, -0.1]).GetLinearInverse()
        cameraToWorld = transformUtils.concatenateTransforms([cameraToTag, tagToWorld])

         #   for objname in kinect_frames_to_handle:
        obj = om.findObjectByName('kinect source frame')
        if obj:
            #filterUtils.transformPolyData(obj, cameraToWorld)
            obj.copyFrame(cameraToWorld)

        '''
        tr = getBotFrame('robot_base')
        base_pos = transformUtils.poseFromTransform(tr)[0]
        base_rpy = transformUtils.rollPitchYawFromTransform(tr)

        robotStateJointController.q[0] = base_pos[0]
        robotStateJointController.q[1] = base_pos[1]
        robotStateJointController.q[2] = base_pos[2]
        robotStateJointController.q[3] = base_rpy[0]
        robotStateJointController.q[4] = base_rpy[1]
        robotStateJointController.q[5] = base_rpy[2]
        robotStateJointController.push()
        '''

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




def frameHandler(msg):
    global kinect_transform_latest

    #Take a windowed average of latest k samples, if available
    kinect_transform_latest.append((msg.trans, msg.quat))
    if len(kinect_transform_latest) > KINECT_TRANSFORM_WINDOW_LENGTH:
        kinect_transform_latest = kinect_transform_latest[-KINECT_TRANSFORM_WINDOW_LENGTH:]

    avg_trans = tuple([sum([kinect_transform_latest[sample][0][ind] \
                        for sample in range(len(kinect_transform_latest))]) / len(kinect_transform_latest) \
                        for ind in range(len(kinect_transform_latest[0][0]))])
    avg_quat = tuple([sum([kinect_transform_latest[sample][1][ind] \
                        for sample in range(len(kinect_transform_latest))]) / len(kinect_transform_latest) \
                        for ind in range(len(kinect_transform_latest[0][1]))])

    botToCamera = transformUtils.transformFromPose(avg_trans, avg_quat)
    cameraToBot = botToCamera.GetLinearInverse()
    botToWorld = transformUtils.frameFromPositionAndRPY(botTranslation, botRpy)

    cameraToWorld = transformUtils.concatenateTransforms([cameraToBot, botToWorld])

#    for objname in to_be_framed:
#        obj = om.findObjectByName(objname)
#        if obj:
#            framename = objname + ' frame'
#            fr = om.findObjectByName(framename)
#            if fr is None:
#                vis.addChildFrame(obj)

 #   for objname in kinect_frames_to_handle:
    obj = om.findObjectByName('kinect source frame')
    if obj:
        #filterUtils.transformPolyData(obj, cameraToWorld)
        obj.copyFrame(cameraToWorld)


tr = getBotFrame('robot_base')
base_pos = transformUtils.poseFromTransform(tr)[0]
base_rpy = transformUtils.rollPitchYawFromTransform(tr)

robotStateJointController.q[0] = base_pos[0]
robotStateJointController.q[1] = base_pos[1]
robotStateJointController.q[2] = base_pos[2]
robotStateJointController.q[3] = base_rpy[0]
robotStateJointController.q[4] = base_rpy[1]
robotStateJointController.q[5] = base_rpy[2]
robotStateJointController.push()


