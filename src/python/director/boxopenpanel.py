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

# Adding from contmanippanel
from director.timercallback import TimerCallback


class BoxPushTaskPlanner(object):

    def __init__(self, robotSystem, manipulandStateModel, manipulandLinkName, manipulatorStateModel, manipulatorLinkName):
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

        #Manipuland
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

        # Grabbing frame Relative to manipuland link
        self.targetLinkToGrabFrame = transformUtils.frameFromPositionAndRPY( [0, 0, 0], [0, 0, 0] )
        # +x is forward for palm frame; Reaching frame relative to grab frame
        self.grabToReachFrame = transformUtils.frameFromPositionAndRPY( [-0.2, 0, 0], [0, 0, 0] )
        # Thumb position relative to manipulator link
        self.eeLinkToHandFrame = transformUtils.frameFromPositionAndRPY( [0, 0, 0.0], [0, 0, 0] )

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
        vis.updateFrame(self.grabFrame, 'BOP_Manipuland Grab Frame', parent='estimation', visible=True, scale=0.2)
        
        grabFrameObj = om.findObjectByName('BOP_Manipuland Grab Frame')
        if grabFrameObj != self.grabFrameObj:
            self.grabFrameObj = grabFrameObj
            self.targetLinkToGrabFrameCallback = self.grabFrameObj.connectFrameModified(self.targetLinkToGrabFrameModified)

        self.reachFrame = transformUtils.copyFrame(self.grabFrame)
        self.reachFrame.PreMultiply()
        self.reachFrame.Concatenate(self.grabToReachFrame)
        vis.updateFrame(self.reachFrame, 'BOP_Manipuland Reach Frame', parent='estimation', visible=True, scale=0.2)
        
        reachFrameObj = om.findObjectByName('BOP_Manipuland Reach Frame')
        if reachFrameObj != self.reachFrameObj:
            self.reachFrameObj = reachFrameObj
            self.grabToReachFrameCallback = self.reachFrameObj.connectFrameModified(self.grabToReachFrameModified)

        self.handFrame = transformUtils.copyFrame(self.manipulatorStateModel.getLinkFrame(self.manipulatorLinkName))
        self.handFrame.PreMultiply()
        self.handFrame.Concatenate(self.eeLinkToHandFrame)
        vis.updateFrame(self.handFrame, 'BOP_Manipulator Hand Frame', parent='estimation', visible=True, scale=0.2)
        
        handFrameObj = om.findObjectByName('BOP_Manipulator Hand Frame')
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
    
    def addPlan(self, plan):
        self.plans.append(plan)
    
    ## Start Box Push Commands 
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
        pose = (array([ 1.20,  0. , 0.8]), array([ 1.,  0.,  0.,  0.]))
        desc = dict(classname='BoxAffordanceItem', Name='box', uuid=newUUID(), pose=pose, Color=[0.66, 0.66, 0.66], Dimensions=[0.25,0.25,0.25])
        obj = self.affordanceManager.newAffordanceFromDescription(desc)

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
        #obj = om.findObjectByName('box')
        #frame = om.findObjectByName('box frame')

        obj = om.findObjectByName('cardboard_box')
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
     ## End Box Push Commands

    def commitManipPlan(self):
        self.manipPlanner.commitManipPlan(self.plans[-1])  
    

    def setupManip(self, task, side='left'):
        # [task_name, manipulator link, hand frame, manipuland link, reach frame, grab frame]
        arm = task[1]     # manipulator link
        handFrame = task[2]    # hand fram relative to manipulator
        mland = task[3]   # manipuiland frame
        touchFrame = task[4]    #reach frame relative to manipuland frame

        # Grabbing frame Relative to manipuland link
        self.targetLinkToGrabFrame = touchFrame
        # +x is forward for palm frame; Reaching frame relative to grab frame
        self.grabToReachFrame = transformUtils.transformFromPose([0,0,0], [1,0,0,0])

        # Thumb position relative to manipulator link
        self.eeLinkToHandFrame = handFrame

        self.manipulatorLinkName = arm
        self.manipulandLinkName = mland
        
        self.update()

        #figure out which frame to grab
        #self.planManipToGivenFrame(self.grabFrame, side)

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
        self.constraintSet = self.ikPlanner.planEndEffectorGoal(startPose, side, endFrame, lockBase=False, lockBack=True, graspToHandLinkFrame=self.eeLinkToHandFrame)
        self.constraintSet.nominalPoseName='q_start'
        self.constraintSet.runIk()
        plan = self.constraintSet.runIkTraj()
        self.addPlan(plan)

class BoxImageFitter(ImageBasedAffordanceFit):

    def __init__(self, boxpushdemo):
        ImageBasedAffordanceFit.__init__(self, numberOfPoints=1)
        self.planner = boxpushdemo

    def fit(self, polyData, points):
        pass

class BoxOpenPanel(TaskUserPanel):

    def __init__(self, robotSystem, manipulandStateModels):

        TaskUserPanel.__init__(self, windowTitle='Example Box Pushing Task')


        self.manipulandStateModels = manipulandStateModels
        self.manipulatorname = 'arm'
        self.manipulandname = 'cardboard_box'

        names = [stateModel.getProperty('Name') for stateModel in manipulandStateModels]

        x=names.index(self.manipulatorname)
        self.manipulator = manipulandStateModels[x]

        x=names.index(self.manipulandname)
        self.manipuland = manipulandStateModels[x]

        # for x in manipulandStateModels.model.getLinkNames()
        #     if str(x) == self.manipulandname
        #         self.manipuland = manipulandStateModels[x]
        #     if str(x) == self.manipulatorname
        #         self.manipulator = manipulandStateModels[x]


        self.updateLinkChoice()

        self.planner = BoxPushTaskPlanner(robotSystem, self.manipuland,
                                  self.manipuland.model.getLinkNames()[0],
                                  self.manipulator,
                                  self.manipulator.model.getLinkNames()[0])
        self.fitter = BoxImageFitter(self.planner)
        self.initImageView(self.fitter.imageView)

        self.addDefaultProperties()
        self.addButtons()
        self.addTasks()

        self.timerCallback = TimerCallback(50)
        self.timerCallback.callback = self.update
        self.timerCallback.start()

    def updateLinkChoice(self):
        manipulandStateModel = self.manipuland
        manipulatorStateModel = self.manipulator
        
    def update(self):
        self.planner.update()
    
    def addButtons(self):
        self.addManualSpacer()
        self.addManualButton('Spawn Box', self.onSpawnBoxClicked)
    
    #Needed to define hand properties for ABB 
    #def addDefaultProperties(self):
    
    # User-defined box    
    def onSpawnBoxClicked(self):
        self.planner.spawnBoxAffordance()

    def testPrint(self):
        self.appendMessage('test')

    def addDefaultProperties(self):
        self.params.addProperty('Hand', 0, attributes=om.PropertyAttributes(enumNames=['Left']))
        self.planner.graspingHand = self.getSide()
        self.planner.planner = self.getPlanner()

    def getSide(self):
        return self.params.getPropertyEnumValue('Hand').lower()

    def getPlanner(self):
        return self.params.getPropertyEnumValue('Planner') if self.params.hasProperty('Planner') else None

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

        def addReach(group):
            addFunc(functools.partial(self.planner.planReach, 'left'), name='plan reach', parent=group)
            addTask(rt.CheckPlanInfo(name='check reach plan info'), parent=group)
            addFunc(self.planner.commitManipPlan, name='execute reach plan', parent=group)
            addTask(rt.IRBWaitForPlanExecution(name='wait for Timeout seconds for manip execution'), parent=group)

        def addGrab(group):
            addFunc(functools.partial(self.planner.planGrab, 'left'), name='plan touch', parent=group)
            addTask(rt.CheckPlanInfo(name='check touch plan info'), parent=group)
            addFunc(self.planner.commitManipPlan, name='execute touch plan', parent=group)
            addTask(rt.IRBWaitForPlanExecution(name='wait for Timeout seconds for manip execution'), parent=group)

        def addManipulation(task, parent=None, confirm=False):
            group = self.taskTree.addGroup(task[0], parent=parent)
            # setup the frames within the planner
            addFunc(functools.partial(self.planner.setupManip, task), name='setup frames', parent=group)
            #addReach(group)
            addGrab(group)

        self.plannedManipFrames = [] # [task_name, manipulator link, hand frame, manipuland link, grab frame]


        self.plannedManipFrames.append( 
            ('prep_touch_lid_ny',
            'link_6', 
            transformUtils.transformFromPose(array([ 0.28045297, -0.0873983 ,  0.00155146]), array([ 0.99776053, -0.06238316,  0.02251967,  0.00866776])),
            'box_lid_ny', 
            transformUtils.transformFromPose(array([-0.14844034,  0.00363105, -0.03874915]), array([ 0.61072786, -0.70531327, -0.01374443, -0.35966062]))
            ) )

        self.plannedManipFrames.append( 
            ('touch_lid_ny',
            'link_6', 
            transformUtils.transformFromPose(array([ 0.28045297, -0.0873983 ,  0.00155146]), array([ 0.99776053, -0.06238316,  0.02251967,  0.00866776])),
            'box_lid_ny', 
            transformUtils.transformFromPose(array([-0.06750864, -0.10518526,  0.01421868]), array([-0.4891737 ,  0.67363407, -0.20124232,  0.51616641])),
            ) )

        self.plannedManipFrames.append( 
            ('slide_under_lid_py',
            'link_6', 
            transformUtils.transformFromPose(array([ 0.28045297, -0.0873983 ,  0.00155146]), array([ 0.99776053, -0.06238316,  0.02251967,  0.00866776])),
            'box_lid_py', 
            transformUtils.transformFromPose(array([-0.07014771,  0.09655235,  0.02164498]), array([-0.49221811,  0.68231605, -0.23736393,  0.48561765]))
            ) )

        self.plannedManipFrames.append( 
            ('push_open_lid_py_1',
            'link_6', 
            transformUtils.transformFromPose(array([ 0.28045297, -0.0873983 ,  0.00155146]), array([ 0.99776053, -0.06238316,  0.02251967,  0.00866776])),
            'box_lid_py', 
            transformUtils.transformFromPose(array([-0.08440133,  0.06643229, -0.06796792]), array([ 0.57253841, -0.62891126,  0.36167245, -0.38192072]))
            ) )

        self.plannedManipFrames.append( 
            ('push_open_lid_py_2',
            'link_6', 
            transformUtils.transformFromPose(array([ 0.28045297, -0.0873983 ,  0.00155146]), array([ 0.99776053, -0.06238316,  0.02251967,  0.00866776])),
            'box_lid_py', 
            transformUtils.transformFromPose(array([-0.04451736,  0.04110726, -0.06619589]), array([ 0.62068669, -0.56271116,  0.47048396, -0.27703617]))
            ) )

        self.plannedManipFrames.append( 
            ('push_open_lid_py_3',
            'link_6', 
            transformUtils.transformFromPose(array([ 0.28045297, -0.0873983 ,  0.00155146]), array([ 0.99776053, -0.06238316,  0.02251967,  0.00866776])),
            'box_lid_py', 
            transformUtils.transformFromPose(array([-0.04643044,  0.04453561, -0.07002755]), array([ 0.66069153, -0.46680209,  0.56662106, -0.15659848]))
            ) )

        self.plannedManipFrames.append( 
            ('push_open_lid_py_4',
            'link_6', 
            transformUtils.transformFromPose(array([ 0.28045297, -0.0873983 ,  0.00155146]), array([ 0.99776053, -0.06238316,  0.02251967,  0.00866776])),
            'box_lid_py', 
            transformUtils.transformFromPose(array([-0.08829975,  0.02615101, -0.08088898]), array([ 0.71597573, -0.13627428,  0.68060439, -0.07473772]))
            ) )

        self.plannedManipFrames.append( 
            ('clear_lid_py_1',
            'link_6', 
            transformUtils.transformFromPose(array([ 0.28045297, -0.0873983 ,  0.00155146]), array([ 0.99776053, -0.06238316,  0.02251967,  0.00866776])),
            'box_wall_ny', 
            transformUtils.transformFromPose(array([-0.1230507 ,  0.37731466,  0.22021508]), array([ 0.71870162, -0.30503559,  0.62298606, -0.04805876]))
            ) )

        self.plannedManipFrames.append( 
            ('clear_lid_py_2',
            'link_6', 
            transformUtils.transformFromPose(array([ 0.28045297, -0.0873983 ,  0.00155146]), array([ 0.99776053, -0.06238316,  0.02251967,  0.00866776])),
            'box_wall_ny', 
            transformUtils.transformFromPose(array([-0.0483845 ,  0.10244286,  0.30024824]), array([ 0.71592305, -0.18180936,  0.66873879,  0.08478196]))
            ) )

        self.plannedManipFrames.append( 
            ('reach_lid_ny',
            'link_6',  # switch to index finger
            transformUtils.transformFromPose(array([ 0.2762586 ,  0.08744795, -0.03422115]), array([ -1.95770323e-02,   9.99808349e-01,  -3.51273567e-05, 5.62755701e-05])),
            'box_wall_ny', 
            transformUtils.transformFromPose(array([-0.05182284,  0.20024828,  0.22180205]), array([ 0.33157685,  0.71718487, -0.0865867 , -0.60679931]))
            ) )

        self.plannedManipFrames.append( 
            ('under_lid_ny',
            'link_6',  # switch to index finger
            transformUtils.transformFromPose(array([ 0.2762586 ,  0.08744795, -0.03422115]), array([ -1.95770323e-02,   9.99808349e-01,  -3.51273567e-05, 5.62755701e-05])),
            'box_lid_ny', 
            transformUtils.transformFromPose(array([-0.06929865, -0.13893348,  0.02239436]), array([ 0.77932306, -0.12096817, -0.61323995, -0.04426089]))
            ) )

        self.plannedManipFrames.append( 
            ('push_open_lid_ny_1',
            'link_6',  # switch to index finger
            transformUtils.transformFromPose(array([ 0.2762586 ,  0.08744795, -0.03422115]), array([ -1.95770323e-02,   9.99808349e-01,  -3.51273567e-05, 5.62755701e-05])),
            'box_lid_ny', 
            transformUtils.transformFromPose(array([-0.0647914 , -0.09641637,  0.00619952]), array([ 0.71487847, -0.37097652, -0.5810347 ,  0.11714893]))
            ) )

        self.plannedManipFrames.append( 
            ('push_open_lid_ny_2',
            'link_6',  # switch to index finger
            transformUtils.transformFromPose(array([ 0.2762586 ,  0.08744795, -0.03422115]), array([ -1.95770323e-02,   9.99808349e-01,  -3.51273567e-05, 5.62755701e-05])),
            'box_lid_ny', 
            transformUtils.transformFromPose(array([-0.06922679, -0.06627821, -0.06163632]), array([ 0.75207569, -0.30828042, -0.53922565,  0.22041107]))
            ) )

        self.plannedManipFrames.append( 
            ('push_open_lid_ny_3',
            'link_6',  # switch to index finger
            transformUtils.transformFromPose(array([ 0.2762586 ,  0.08744795, -0.03422115]), array([ -1.95770323e-02,   9.99808349e-01,  -3.51273567e-05, 5.62755701e-05])),
            'box_lid_ny', 
            transformUtils.transformFromPose(array([-0.06831325, -0.04608639, -0.07934433]), array([ 0.60479311, -0.49542224, -0.37648556,  0.49703192]))
            ) )

        self.plannedManipFrames.append( 
            ('push_open_lid_ny_4',
            'link_6',  # switch to index finger
            transformUtils.transformFromPose(array([ 0.2762586 ,  0.08744795, -0.03422115]), array([ -1.95770323e-02,   9.99808349e-01,  -3.51273567e-05, 5.62755701e-05])),
            'box_lid_ny', 
            transformUtils.transformFromPose(array([-0.04805165, -0.03240205, -0.0680926 ]), array([ 0.1980355 , -0.65634955, -0.00909902,  0.72794534]))
            ) )

        self.plannedManipFrames.append( 
            ('grasp_prep_1',
            'link_6',  # switch to index finger
            transformUtils.transformFromPose(array([ 0.2762586 ,  0.08744795, -0.03422115]), array([ -1.95770323e-02,   9.99808349e-01,  -3.51273567e-05, 5.62755701e-05])),
            'box_wall_ny', 
            transformUtils.transformFromPose(array([-0.07064733,  0.12626073,  0.30145889]), array([ 0.02972969,  0.72072932,  0.03412298, -0.69173768]))
            ) )

        self.plannedManipFrames.append( 
            ('grasp_prep_2',
            'link_6',  # switch to index finger
            transformUtils.transformFromPose(array([ 0.26640564,  0.00887525, -0.00600118]), array([ -1.62187647e-03,   9.99998681e-01,  -3.24007632e-05, 8.57870447e-05])),
            'box_wall_ny', 
            transformUtils.transformFromPose(array([-0.07072407,  0.11885632,  0.27238985]), array([ 0.02749294,  0.722208  ,  0.0323756 , -0.6903706 ]))
            ) )

        self.taskTree.removeAllTasks()

        for task in self.plannedManipFrames:
            addManipulation(task)

        # TODO: bind these to buttons
        #if self.planner.ikPlanner.fixedBaseArm:
            #addReach('Extra')
            #addGrab('Extra')
            
     #    ############
     #    # some helpers
     #    self.folder = None
     #    def addTask(task, parent=None):
     #        parent = parent or self.folder
     #        self.taskTree.onAddTask(task, copy=False, parent=parent)
     #    def addFunc(func, name, parent=None):
     #        addTask(rt.CallbackTask(callback=func, name=name), parent=parent)
     #    def addFolder(name, parent=None):
     #        self.folder = self.taskTree.addGroup(name, parent=parent)
     #        return self.folder

     #    #Additions from tabledemo.py     
     #    def addGrasping(mode, name, parent=None, confirm=False):
     #        assert mode in ('open', 'close')
     #        group = self.taskTree.addGroup(name, parent=parent)
     #        side = self.params.getPropertyEnumValue('Hand')

     #        checkStatus = False  # whether to confirm that there is an object in the hand when closed

     #        if mode == 'open':
     #            addTask(rt.OpenHand(name='open grasp hand', side=side, CheckStatus=checkStatus), parent=group)
     #        else:
     #            addTask(rt.CloseHand(name='close grasp hand', side=side, CheckStatus=checkStatus), parent=group)
     #        if confirm:
     #            addTask(rt.UserPromptTask(name='Confirm grasping has succeeded', message='Continue when grasp finishes.'),
     #                    parent=group)

     #    def addManipulation(func, name, parent=None, confirm=False):
     #        group = self.taskTree.addGroup(name, parent=parent)
     #        addFunc(func, name='plan motion', parent=group)
     #        addTask(rt.CheckPlanInfo(name='check manip plan info'), parent=group)
     #        addFunc(v.commitManipPlan, name='execute manip plan', parent=group)
     #        if self.planner.planner != 1:
     #            addTask(rt.IRBWaitForPlanExecution(name='wait for Timeout seconds for manip execution'), parent=group)
     #            #if confirm:
     #              # addTask(rt.UserPromptTask(name='Confirm execution has finished', message='Continue when plan finishes.'), parent=group)

     #    self.taskTree.removeAllTasks()

     #    ###############
     #    v = self.planner

     #    # graspingHand is 'left', side is 'Left'
     #    side = self.params.getPropertyEnumValue('Hand')

     #    # add the tasks

	    # # fit
     #    fit = self.taskTree.addGroup('Fitting')
     #    addTask(rt.UserPromptTask(name='fit box',
     #                              message='Please fit and approve box affordance.'), parent=fit)
     #    addTask(rt.FindAffordance(name='check box affordance', affordanceName='box'),
     #            parent=fit)

     #    # lift object
     #    if v.ikPlanner.fixedBaseArm:
     #        addManipulation(functools.partial(v.planPreGrasp, v.graspingHand ), name='raise arm')
     #        addManipulation(functools.partial(v.planReachToTableObject, v.graspingHand), name='reach')
     #        addManipulation(functools.partial(v.planTouchTableObject, v.graspingHand), name='touch')




