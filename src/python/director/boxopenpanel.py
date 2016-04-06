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

import os
import sys
import vtkAll as vtk
import numpy as np

##Adding from tabedemo
import functools
import drcargs
from numpy import array
from director.uuidutil import newUUID
from director import affordanceitems
from director import segmentation
from director import affordanceupdater
from director import robotstate


class BoxPushTaskPlanner(object):

    def __init__(self, robotSystem):
        self.robotSystem = robotSystem
        self.robotModel = robotSystem.robotStateModel
        self.ikPlanner = robotSystem.ikPlanner
        self.manipPlanner = robotSystem.manipPlanner
        
        #Need for affordance manager.
        self.affordanceManager = segmentation.affordanceManager
        
        #Needed for plans
        self.plans = []
        self.sensorJointController = robotSystem.robotStateJointController

        # For positions arm relattive to box
        self.reachDist = 0.09
        self.view = robotSystem.view


    def test(self):
        print 'test'
        assert True

    
    def addPlan(self, plan):
        self.plans.append(plan)

    def computeRobotStanceFrame(self, objectTransform, relativeStanceTransform):
        '''
        Given a robot model, determine the height of the ground using an XY and
        Yaw standoff, combined to determine the relative 6DOF standoff For a
        grasp or approach stance
        '''

        groundFrame = self.footstepPlanner.getFeetMidPoint(self.robotModel)
        groundHeight = groundFrame.GetPosition()[2]

        graspPosition = np.array(objectTransform.GetPosition())
        graspYAxis = [0.0, 1.0, 0.0]
        graspZAxis = [0.0, 0.0, 1.0]
        objectTransform.TransformVector(graspYAxis, graspYAxis)
        objectTransform.TransformVector(graspZAxis, graspZAxis)

        xaxis = graspYAxis
        zaxis = [0, 0, 1]
        yaxis = np.cross(zaxis, xaxis)
        yaxis /= np.linalg.norm(yaxis)
        xaxis = np.cross(yaxis, zaxis)

        graspGroundTransform = transformUtils.getTransformFromAxes(xaxis, yaxis, zaxis)
        graspGroundTransform.PostMultiply()
        graspGroundTransform.Translate(graspPosition[0], graspPosition[1], groundHeight)

        robotStance = transformUtils.copyFrame(relativeStanceTransform)
        robotStance.Concatenate(graspGroundTransform)

        return robotStance

    def computeRelativeGraspTransform(self):
        t = transformUtils.copyFrame(transformUtils.frameFromPositionAndRPY(self.graspFrameXYZ,
                                                                            self.graspFrameRPY))
        t.PostMultiply()
        t.RotateX(180)
        t.RotateY(-90)
        return t    

    def computeRelativeStanceTransform(self):
        return transformUtils.copyFrame(
            transformUtils.frameFromPositionAndRPY(self.relativeStanceXYZ, self.relativeStanceRPY))    

    def spawnBoxAffordance(self):
        # radius = 0.20
        # tubeRadius = 0.02
        # position = [0, 0, 1.2]
        # rpy = [0, 0, 0]
        # t_feet_mid = self.footstepPlanner.getFeetMidPoint(self.robotModel)
        # t = transformUtils.frameFromPositionAndRPY(position, rpy)
        # t_grasp = self.computeRelativeGraspTransform()
        # t_grasp.Concatenate(t)
        # t_stance = self.computeRobotStanceFrame(t_grasp, self.computeRelativeStanceTransform())
        # t_valve = t_stance.GetInverse()

        # # This is necessary to get the inversion to actually happen. We don't know why.
        # t_valve.GetMatrix()

        # t_valve.Concatenate(t)
        # t_valve.Concatenate(t_feet_mid)
        # pose = transformUtils.poseFromTransform(t_valve)

        #TODO This should look for a box iten not valve or capsule ring, but dont know how to do this yet
        pose = (array([ 1.20,  0. , 0.8]), array([ 1.,  0.,  0.,  0.]))
        desc = dict(classname='BoxAffordanceItem', Name='box', uuid=newUUID(), pose=pose, Color=[0.66, 0.66, 0.66], Dimensions=[0.25,0.25,0.25])
        obj = self.affordanceManager.newAffordanceFromDescription(desc)

        #TODO Shoudl look for a box now from desc; need to give params of sides instead of radius
        # obj = segmentation.affordanceManager.newAffordanceFromDescription(desc)
        # obj.params = dict(radius=radius)
    
    def getEstimatedRobotStatePose(self):
        return self.sensorJointController.getPose('EST_ROBOT_STATE')

    def getPlanningStartPose(self):
        #if self.planFromCurrentRobotState:
            return self.getEstimatedRobotStatePose()
        #else:
        #    if self.plans:
        #        return robotstate.convertStateMessageToDrakePose(self.plans[-1].plan[-1])
        #    else:
        #       return self.getEstimatedRobotStatePose()
        #return robotstate.convertStateMessageToDrakePose(self.plans[-1].plan[-1])

    def planPreGrasp(self, side='left'):
        startPose = self.getPlanningStartPose()
        endPose = self.ikPlanner.getMergedPostureFromDatabase(startPose, 'General', 'Object Held', side=side)  #om.findObjectByName('box frame')
        newPlan = self.ikPlanner.computePostureGoal(startPose, endPose) ## embedded call in IK planner
        self.addPlan(newPlan)

    def planReachToTableObject(self, side='left'):

        #TODO change this to box object pose! not next object
        #obj, frame = self.getNextTableObject(side)
        obj = om.findObjectByName('box')
        frame = om.findObjectByName('box frame')

        startPose = self.getPlanningStartPose()

        #if self.ikPlanner.fixedBaseArm: # includes reachDist hack instead of in ikPlanner (TODO!)
        f = transformUtils.frameFromPositionAndRPY( np.array(frame.transform.GetPosition())-np.array([0.0,self.reachDist+.15,-.03]), [0,0,-90] )
        f.PreMultiply()
        f.RotateY(90)
        f.Update()
        self.constraintSet = self.ikPlanner.planEndEffectorGoal(startPose, side, f, lockBase=False, lockBack=True)
        #newFrame = vis.FrameItem('reach_item', f, self.view)
        #self.constraintSet = self.ikPlanner.planGraspOrbitReachPlan(startPose, side, newFrame, constraints=None, dist=self.reachDist, lockBase=self.lockBase, lockBack=self.lockBack, lockArm=False)
       
        self.constraintSet.runIk()

        print 'planning touch'
        plan = self.constraintSet.runIkTraj()
        self.addPlan(plan)

    def planTouchTableObject(self, side='left'):

        #TODO change this to box object pose! not next object
        #obj, frame = self.getNextTableObject(side)
        obj = om.findObjectByName('box')
        frame = om.findObjectByName('box frame')
        
        startPose = self.getPlanningStartPose()

        #if self.ikPlanner.fixedBaseArm: # includes distance hack and currently uses reachDist instead of touchDist (TODO!)
        f = transformUtils.frameFromPositionAndRPY( np.array(frame.transform.GetPosition())-np.array([0.0,self.reachDist-.15,-.03]), [0,0,-90] )
        f.PreMultiply()
        f.RotateY(90)
        f.Update()
        item = vis.FrameItem('reach_item', f, self.view)
        self.constraintSet = self.ikPlanner.planEndEffectorGoal(startPose, side, f, lockBase=False, lockBack=True)
        # else: # for non-ABB arm
        #     self.constraintSet = self.ikPlanner.planGraspOrbitReachPlan(startPose, side, frame, dist=0.05, lockBase=self.lockBase, lockBack=self.lockBack)
        #     self.constraintSet.constraints[-1].tspan = [-np.inf, np.inf]
        #     self.constraintSet.constraints[-2].tspan = [-np.inf, np.inf]
        
        self.constraintSet.runIk()

        print 'planning touch'
        plan = self.constraintSet.runIkTraj()
        self.addPlan(plan)

    def commitManipPlan(self):
        self.manipPlanner.commitManipPlan(self.plans[-1])    
   

class BoxImageFitter(ImageBasedAffordanceFit):

    def __init__(self, boxpushdemo):
        ImageBasedAffordanceFit.__init__(self, numberOfPoints=1)
        self.boxpushdemo = boxpushdemo

    def fit(self, polyData, points):
        pass

# class ValveImageFitter(ImageBasedAffordanceFit):

#     def __init__(self, valveDemo):
#         ImageBasedAffordanceFit.__init__(self, numberOfPoints=2)
#         self.valveDemo = valveDemo

#     def onImageViewDoubleClick(self, displayPoint, modifiers, imageView):
#         self.valveDemo.onImageViewDoubleClick(displayPoint, modifiers, imageView)

#     def fit(self, polyData, points):
#         om.removeFromObjectModel(om.findObjectByName('valve'))
#         segmentation.segmentValveByRim(polyData, points[0], points[1])

class BoxOpenPanel(TaskUserPanel):

    def __init__(self, robotSystem):

        TaskUserPanel.__init__(self, windowTitle='Example Box Pushing Task')

        self.boxpushdemo = BoxPushTaskPlanner(robotSystem)
        self.fitter = BoxImageFitter(self.boxpushdemo)
        self.initImageView(self.fitter.imageView)

        self.addDefaultProperties()
        self.addButtons()
        self.addTasks()

    def addButtons(self):
        self.addManualButton('box button', self.boxpushdemo.test)
        self.addManualSpacer()
        self.addManualButton('Spawn Box', self.onSpawnBoxClicked)
    
    #Needed to define hand properties for ABB 
    def addDefaultProperties(self):
        # TODO: if there is a way not to display a property where there is only one value, that'd be great

            # Hard-coded left for now per convention, can also be generic per drcargs.getDirectorConfig()['handCombinations']['side']
        self.params.addProperty('Hand', 0,
                                attributes=om.PropertyAttributes(enumNames=['Left']))

        #if self.tableDemo.ikPlanner.fixedBaseArm:
        self.params.addProperty('Base', 0,
                                attributes=om.PropertyAttributes(enumNames=['Fixed']))
        self.params.addProperty('Back', 0,
                                attributes=om.PropertyAttributes(enumNames=['Fixed']))

        # Hand control for Kuka LWR / Schunk SDH
        if 'userConfig' in drcargs.getDirectorConfig() and 'useKuka' in drcargs.getDirectorConfig()['userConfig']:
            self.params.addProperty('Hand Engaged (Powered)', False)

        # Init values as above
        self.boxpushdemo.graspingHand = self.getSide()
        self.boxpushdemo.lockBase = self.getLockBase()
        self.boxpushdemo.lockBack = self.getLockBack()
        self.boxpushdemo.sceneID = self.getSceneId()
        self.boxpushdemo.sceneName = self.getSceneName()
        self.boxpushdemo.planner = self.getPlanner()
        if 'userConfig' in drcargs.getDirectorConfig() and 'useKuka' in drcargs.getDirectorConfig()['userConfig']:
            self.handEngaged = self.getHandEngaged() # WARNING: does not check current state [no status message]    

    def getSide(self):
        return self.params.getPropertyEnumValue('Hand').lower()

    def getLockBase(self):
        return True if self.params.getPropertyEnumValue('Base') == 'Fixed' else False

    def getLockBack(self):
        return True if self.params.getPropertyEnumValue('Back') == 'Fixed' else False

    def getHandEngaged(self):
        return self.params.getProperty('Hand Engaged (Powered)')

    def getSceneId(self):
        return self.params.getProperty('Scene') if self.params.hasProperty('Scene') else None

    def getSceneName(self):
        return self.params.getPropertyEnumValue('Scene') if self.params.hasProperty('Scene') else None

    def getPlanner(self):
        return self.params.getPropertyEnumValue('Planner') if self.params.hasProperty('Planner') else None
    #    
    def onSpawnBoxClicked(self):
        self.boxpushdemo.spawnBoxAffordance()

    def testPrint(self):
        self.appendMessage('test')

    # def addDefaultProperties(self):
    #     self.params.addProperty('Example bool', True)
    #     self.params.addProperty('Example enum', 0, attributes=om.PropertyAttributes(enumNames=['Left', 'Right']))
    #     self.params.addProperty('Example double', 1.0, attributes=om.PropertyAttributes(singleStep=0.01, decimals=3))

    def onPropertyChanged(self, propertySet, propertyName):
          self.appendMessage('property changed: <b>%s</b>' % propertyName)
          self.appendMessage('  new value: %r' % self.params.getProperty(propertyName))

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

        #Additions from tabledemo.py     
        def addGrasping(mode, name, parent=None, confirm=False):
            assert mode in ('open', 'close')
            group = self.taskTree.addGroup(name, parent=parent)
            side = self.params.getPropertyEnumValue('Hand')

            checkStatus = False  # whether to confirm that there is an object in the hand when closed
            if 'userConfig' in drcargs.getDirectorConfig() and 'useKuka' in drcargs.getDirectorConfig()['userConfig']:
                checkStatus = True

            if mode == 'open':
                addTask(rt.OpenHand(name='open grasp hand', side=side, CheckStatus=checkStatus), parent=group)
            else:
                addTask(rt.CloseHand(name='close grasp hand', side=side, CheckStatus=checkStatus), parent=group)
            if confirm:
                addTask(rt.UserPromptTask(name='Confirm grasping has succeeded', message='Continue when grasp finishes.'),
                        parent=group)

        def addManipulation(func, name, parent=None, confirm=False):
            group = self.taskTree.addGroup(name, parent=parent)
            addFunc(func, name='plan motion', parent=group)
            addTask(rt.CheckPlanInfo(name='check manip plan info'), parent=group)
            addFunc(v.commitManipPlan, name='execute manip plan', parent=group)
            if self.boxpushdemo.planner != 1:
                addTask(rt.IRBWaitForPlanExecution(name='wait for Timeout seconds for manip execution'), parent=group)
                #if confirm:
                  # addTask(rt.UserPromptTask(name='Confirm execution has finished', message='Continue when plan finishes.'), parent=group)

        self.taskTree.removeAllTasks()
        ###############
        v = self.boxpushdemo

        # graspingHand is 'left', side is 'Left'
        side = self.params.getPropertyEnumValue('Hand')

        # add the tasks

	    # fit
        fit = self.taskTree.addGroup('Fitting')
        addTask(rt.UserPromptTask(name='fit box',
                                  message='Please fit and approve box affordance.'), parent=fit)
        addTask(rt.FindAffordance(name='check box affordance', affordanceName='box'),
                parent=fit)

        # lift object
        if v.ikPlanner.fixedBaseArm:
            addManipulation(functools.partial(v.planPreGrasp, v.graspingHand ), name='raise arm')
            addManipulation(functools.partial(v.planReachToTableObject, v.graspingHand), name='reach')
            addManipulation(functools.partial(v.planTouchTableObject, v.graspingHand), name='touch')

        #tasks =self.taskTree.addGroup('Tasks')
        #addTask(rt.PrintTask(name='display message', message='hello world!@'))
        #addTask(rt.DelayTask(name='wait', delayTime=1.0))
        #addTask(rt.UserPromptTask(name='prompt for user input', message='please press continue...'))
        #addFunc(self.planner.test, name='test planner')
        #addGrasping('open', name='open hand', parent='tasks', confirm=False)




