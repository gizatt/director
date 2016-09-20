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
import functools
import sys
import vtkAll as vtk
import numpy as np



class ContinuousManipPlanner(object):

    def __init__(self, robotSystem, manipulandStateModel, manipulandLinkName, manipulatorStateModel, manipulatorLinkName):
        self.robotSystem = robotSystem
        self.robotModel = robotSystem.robotStateModel
        self.ikPlanner = robotSystem.ikPlanner
        self.sensorJointController = robotSystem.robotStateJointController
        self.manipPlanner = robotSystem.manipPlanner
        self.manipulandStateModel = manipulandStateModel
        self.manipulandLinkName = manipulandLinkName
        self.manipulatorStateModel = manipulatorStateModel
        self.manipulatorLinkName = manipulatorLinkName

        self.plans = []

        self.grabFrameObj = None
        self.reachFrameObj = None
        self.handFrameObj = None
        self.grabFrame = None
        self.reachFrame = None
        self.handFrame = None

        self.resetHand()
        self.resetReachAndGrab()

    def resetHand(self):
        # +x is forward for palm frame
        self.eeLinkToHandFrame = transformUtils.frameFromPositionAndRPY( [0, 0, 0.0], [0, 0, 0] )
        
    def resetReachAndGrab(self):
        self.grabToReachFrame = transformUtils.frameFromPositionAndRPY( [-0.05, 0, 0], [0, 0, 0] )
        self.targetLinkToGrabFrame = transformUtils.frameFromPositionAndRPY( [0, 0, -0.295], [0, -90, 0] )

    def targetLinkToGrabFrameModified(self, grabFrameObj):
        self.targetLinkToGrabFrame = transformUtils.copyFrame(grabFrameObj.transform)
        self.targetLinkToGrabFrame.PostMultiply()
        self.targetLinkToGrabFrame.Concatenate(self.manipulandStateModel.getLinkFrame(self.manipulandLinkName).GetLinearInverse())

    def grabToReachFrameModified(self, reachFrameObj):
        self.grabToReachFrame = transformUtils.copyFrame(reachFrameObj.transform)
        self.grabToReachFrame.Concatenate(self.manipulandStateModel.getLinkFrame(self.manipulandLinkName).GetLinearInverse())
        self.grabToReachFrame.Concatenate(self.targetLinkToGrabFrame.GetLinearInverse())

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
        # self.manipulandStateModel.getLinkFrame('tiny_screwdriver_body') ### get link transform to for current model pose

    def addPlan(self, plan):
        self.plans.append(plan)

    def commitManipPlan(self):
        self.manipPlanner.commitManipPlan(self.plans[-1])    

    def planReach(self, side='left'):
        self.planManipToGivenFrame(self.reachFrame, side)

    def planGrab(self, side='left'):
        self.planManipToGivenFrame(self.grabFrame, side)
        
    def planManipToGivenFrame(self, targetFrame, side='left'):
        startPose = self.sensorJointController.getPose('EST_ROBOT_STATE') # ground truth start pose

        startFrame = transformUtils.copyFrame(self.ikPlanner.getLinkFrameAtPose(self.manipulatorLinkName, startPose))
        startFrame.PreMultiply()
        startFrame.Concatenate(self.eeLinkToHandFrame)
        #vis.updateFrame(startFrame, 'Start Frame', parent='estimation', visible=True, scale=0.2)

        # calculate desired end pose by calculating difference between hand and target, and
        # adding that to start pose
        eeFrameError = transformUtils.copyFrame(targetFrame)
        eeFrameError.PreMultiply()
        eeFrameError.Concatenate(self.handFrame.GetLinearInverse())
        #vis.updateFrame(eeFrameError, 'Error Frame', parent='estimation', visible=True, scale=0.2)

        endFrame = transformUtils.copyFrame(startFrame)
        endFrame.PostMultiply()
        endFrame.Concatenate(eeFrameError)
        #vis.updateFrame(endFrame, 'Target Frame', parent='estimation', visible=True, scale=0.2)

        # eeLinkToHandFrame replaces the handlink -> palm transform
        self.constraintSet = self.ikPlanner.planLinkGoal(startPose, side, self.manipulatorLinkName, endFrame, self.eeLinkToHandFrame, lockBase=False, lockBack=True)

        self.constraintSet.runIk()
        plan = self.constraintSet.runIkTraj()
        self.addPlan(plan)

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

        self.addTasks()
        self.addButtons()

        self.timerCallback = TimerCallback(100)
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
        self.addManualButton('resetReachAndGrab', self.planner.resetReachAndGrab)
        self.addManualButton('resetHand', self.planner.resetHand)

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

        def addManipulation(func, name, parent=None, confirm=False):
            group = self.taskTree.addGroup(name, parent=parent)
            addFunc(func, name='plan motion', parent=group)
            addTask(rt.CheckPlanInfo(name='check manip plan info'), parent=group)
            addFunc(self.planner.commitManipPlan, name='execute manip plan', parent=group)
            addTask(rt.IRBWaitForPlanExecution(name='wait for Timeout seconds for manip execution'), parent=group)
                #if confirm:
                  # addTask(rt.UserPromptTask(name='Confirm execution has finished', message='Continue when plan finishes.'), parent=group)

        self.taskTree.removeAllTasks()


        # add the tasks
        if self.planner.ikPlanner.fixedBaseArm:
            addManipulation(functools.partial(self.planner.planReach, 'left'), name='reach')
            addManipulation(functools.partial(self.planner.planGrab, 'left'), name='grab')



