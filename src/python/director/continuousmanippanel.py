from director import lcmUtils
from director import transformUtils
from director import objectmodel as om
from director import visualization as vis
from director.debugVis import DebugData
from director import ikplanner
from director.ikparameters import IkParameters
from director import vtkNumpy as vnp
from director.tasks.taskuserpanel import TaskUserPanel
from director.tasks.taskuserpanel import ImageBasedAffordanceFit
import director.tasks.robottasks as rt
from director.timercallback import TimerCallback

import os
import sys
import vtkAll as vtk
import numpy as np



class ContinuousManipPlanner(object):

    def __init__(self, robotSystem, manipulandStateModel, manipulandLinkName, manipulatorStateModel, manipulatorLinkName):
        self.robotSystem = robotSystem
        self.robotModel = robotSystem.robotStateModel
        self.ikPlanner = robotSystem.ikPlanner
        self.manipulandStateModel = manipulandStateModel
        self.manipulandLinkName = manipulandLinkName
        self.manipulatorStateModel = manipulatorStateModel
        self.manipulatorLinkName = manipulatorLinkName

        self.grabFrameObj = None
        self.reachFrameObj = None
        self.handFrameObj = None
        self.grabFrame = None
        self.reachFrame = None
        self.handFrame = None

        self.targetLinkToGrabFrame = transformUtils.frameFromPositionAndRPY( [0, 0, 0], [0, 0, 0] )
        self.grabToReachFrame = transformUtils.frameFromPositionAndRPY( [0, 0, 0.2], [0, 0, 0] )
        self.eeLinkToHandFrame = transformUtils.frameFromPositionAndRPY( [0, 0, 0.2], [0, 0, 0] )

    def targetLinkToGrabFrameModified(self, grabFrameObj):
        self.targetLinkToGrabFrame = transformUtils.copyFrame(grabFrameObj.transform)
        self.targetLinkToGrabFrame.Concatenate(self.manipulandStateModel.getLinkFrame(self.manipulandLinkName).GetLinearInverse())

    def grabToReachFrameModified(self, reachFrameObj):
        self.grabToReachFrame = transformUtils.copyFrame(reachFrameObj.transform)
        self.grabToReachFrame.Concatenate(self.targetLinkToGrabFrame.GetLinearInverse())
        self.grabToReachFrame.Concatenate(self.manipulandStateModel.getLinkFrame(self.manipulandLinkName).GetLinearInverse())

    def eeLinkToHandFrameModified(self, handFrameObj):
        self.eeLinkToHandFrame = transformUtils.copyFrame(handFrameObj.transform)
        self.eeLinkToHandFrame.Concatenate(self.manipulatorStateModel.getLinkFrame(self.manipulatorLinkName).GetLinearInverse())

    def update(self):
        self.grabFrame = transformUtils.copyFrame( self.manipulandStateModel.getLinkFrame(self.manipulandLinkName) )
        self.grabFrame.PreMultiply()
        self.grabFrame.Concatenate( self.targetLinkToGrabFrame )
        vis.updateFrame(self.grabFrame, 'Manipuland Grab Frame', parent='estimation', visible=True, scale=0.2)
        
        grabFrameObj = om.findObjectByName('Manipuland Grab Frame')
        if grabFrameObj != self.grabFrameObj:
            self.grabFrameObj = grabFrameObj
            self.targetLinkToGrabFrameCallback = self.grabFrameObj.connectFrameModified(self.targetLinkToGrabFrameModified)

        self.reachFrame = transformUtils.copyFrame(self.grabFrame)
        self.reachFrame.PreMultiply()
        self.reachFrame.Concatenate(self.grabToReachFrame)
        vis.updateFrame(self.reachFrame, 'Manipuland Reach Frame', parent='estimation', visible=True, scale=0.2)
        
        reachFrameObj = om.findObjectByName('Manipuland Reach Frame')
        if reachFrameObj != self.reachFrameObj:
            self.reachFrameObj = reachFrameObj
            self.grabToReachFrameCallback = self.reachFrameObj.connectFrameModified(self.grabToReachFrameModified)

        self.handFrame = transformUtils.copyFrame(self.manipulatorStateModel.getLinkFrame(self.manipulatorLinkName))
        self.handFrame.PreMultiply()
        self.handFrame.Concatenate(self.eeLinkToHandFrame)
        vis.updateFrame(self.handFrame, 'Manipulator Hand Frame', parent='estimation', visible=True, scale=0.2)
        
        handFrameObj = om.findObjectByName('Manipulator Hand Frame')
        if handFrameObj != self.handFrameObj:
            self.handFrameObj = handFrameObj
            self.eeLinkToHandFrameCallback = self.handFrameObj.connectFrameModified(self.eeLinkToHandFrameModified)

    def updateModels(self, manipulandStateModel, manipulandLinkName, manipulatorStateModel, manipulatorLinkName):
        if self.manipulandStateModel is not manipulandStateModel:
            self.manipulandStateModel = manipulandStateModel
        if self.manipulandLinkName is not manipulandLinkName:
            self.manipulandLinkName = manipulandLinkName
        if self.manipulatorStateModel is not manipulatorStateModel:
            self.manipulatorStateModel = manipulatorStateModel
        if self.manipulatorLinkName is not manipulatorLinkName:
            self.manipulatorLinkName = manipulatorLinkName

class ContinuousManipPanel(TaskUserPanel):

    def __init__(self, robotSystem, manipulandStateModels):

        TaskUserPanel.__init__(self, windowTitle='Continuous Manip')

        self.manipulandStateModels = manipulandStateModels
        availableManipulandNames = [stateModel.getProperty('Name') for stateModel in manipulandStateModels]
        self.params.addProperty('Manipuland Name', 0, attributes=om.PropertyAttributes(enumNames=availableManipulandNames))
        self.params.addProperty('Manipulator Name', 1, attributes=om.PropertyAttributes(enumNames=availableManipulandNames))
        self.updateLinkChoice()

        self.planner = ContinuousManipPlanner(robotSystem, self.manipulandStateModels[self.params.getProperty('Manipuland Name')],
                                  self.params.getPropertyEnumValue('Manipuland Link'),
                                  self.manipulandStateModels[self.params.getProperty('Manipulator Name')],
                                  self.params.getPropertyEnumValue('Manipulator Link'))

        self.timerCallback = TimerCallback(10)
        self.timerCallback.callback = self.update
        self.timerCallback.start()

    def updateLinkChoice(self):
        manipulandStateModel = self.manipulandStateModels[self.params.getProperty('Manipuland Name')]
        manipulatorStateModel = self.manipulandStateModels[self.params.getProperty('Manipulator Name')]
        if not self.params.hasProperty('Manipuland Link'):
            self.params.addProperty('Manipuland Link', 0, attributes=om.PropertyAttributes(enumNames=[str(x) for x in manipulandStateModel.model.getLinkNames()]))
        else:
            self.params.setProperty('Manipuland Link', 0)
            self.params.setPropertyAttribute('Manipuland Link', 'enumNames', [str(x) for x in manipulandStateModel.model.getLinkNames()])

        if not self.params.hasProperty('Manipulator Link'):
            self.params.addProperty('Manipulator Link', 0, attributes=om.PropertyAttributes(enumNames=[str(x) for x in manipulatorStateModel.model.getLinkNames()]))
        else:
            self.params.setProperty('Manipulator Link', 0)
            self.params.setPropertyAttribute('Manipulator Link', 'enumNames', [str(x) for x in manipulatorStateModel.model.getLinkNames()])

    def onPropertyChanged(self, propertySet, propertyName):
        self.appendMessage('property changed: <b>%s</b>' % propertyName)
        self.appendMessage('  new value: %r' % self.params.getProperty(propertyName))
        if (propertyName == 'Manipuland Name'):
            self.updateLinkChoice()
        elif (propertyName == 'Manipulator Name'):
            self.updateLinkChoice()
        elif (propertyName == 'Manipuland Link'):
            self.appendMessage("New target link")
        elif (propertyName == 'Manipulator Link'):
            self.appendMessage("New EE link")
        else:
            self.appendMessage("unknown property changed!?")

        self.planner.updateModels(self.manipulandStateModels[self.params.getProperty('Manipuland Name')],
                                  self.params.getPropertyEnumValue('Manipuland Link'),
                                  self.manipulandStateModels[self.params.getProperty('Manipulator Name')],
                                  self.params.getPropertyEnumValue('Manipulator Link'))

    def update(self):
        self.planner.update()

    def addButtons(self):
        self.addManualButton('example button', self.planner.test)

    def testPrint(self):
        self.appendMessage('test')

    def addDefaultProperties(self):
        self.params.addProperty('Example bool', True)
        self.params.addProperty('Example enum', 0, attributes=om.PropertyAttributes(enumNames=['Left', 'Right']))
        self.params.addProperty('Example double', 1.0, attributes=om.PropertyAttributes(singleStep=0.01, decimals=3))


    def addTasks(self):

        ############
        # some helpers
        self.folder = None
        def addTask(task, parent=None):
            parent = parent or self.folder
            self.taskTree.onAddTask(task, copy=False, parent=parent)
        def addFunc(func, name, parent=None):
            addTask(rt.CallbackTask(callback=func, name=name), parent=parent)
        def addFolder(name, parent=None):
            self.folder = self.taskTree.addGroup(name, parent=parent)
            return self.folder

        self.taskTree.removeAllTasks()
        ###############


        # add the tasks

        addFolder('Pre-grasp')
        addTask(rt.PrintTask(name='display message', message='hello world'))
        addTask(rt.DelayTask(name='wait', delayTime=1.0))
        addTask(rt.UserPromptTask(name='prompt for user input', message='please press continue...'))
        addFunc(self.planner.test, name='test planner')



