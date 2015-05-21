#import ddapp
from ddapp import cameraview
from ddapp import transformUtils
from ddapp import visualization as vis
from ddapp import objectmodel as om
from ddapp import ik
from ddapp.ikparameters import IkParameters
from ddapp.ikplanner import ConstraintSet
from ddapp import polarisplatformplanner
from ddapp import robotstate
from ddapp import segmentation
from ddapp import sitstandplanner
from ddapp.timercallback import TimerCallback
from ddapp import visualization as vis
from ddapp import planplayback
from ddapp import lcmUtils
from ddapp import affordancepanel
from ddapp.uuidutil import newUUID

import os
import functools
import numpy as np
import scipy.io
import vtkAll as vtk
import bot_core as lcmbotcore
from ddapp.tasks.taskuserpanel import TaskUserPanel
import ddapp.tasks.robottasks as rt
from ddapp import filterUtils
from ddapp import ioUtils


class PolarisModel(object):

    def __init__(self):
        self.aprilTagSubsciber = lcmUtils.addSubscriber('APRIL_TAG_TO_CAMERA_LEFT', lcmbotcore.rigid_transform_t, self.onAprilTag)
        pose = transformUtils.poseFromTransform(vtk.vtkTransform())
        desc = dict(classname='MeshAffordanceItem', Name='polaris',
                    Filename='software/models/polaris/polaris_cropped.vtp', pose=pose)
        self.pointcloudAffordance = segmentation.affordanceManager.newAffordanceFromDescription(desc)
        self.originFrame = self.pointcloudAffordance.getChildFrame()
        self.originToAprilTransform = transformUtils.transformFromPose(np.array([-0.038508  , -0.00282131, -0.01000079]), 
            np.array([  9.99997498e-01,  -2.10472556e-03,  -1.33815696e-04, 7.46246794e-04])) # offset for  . . . who knows why

        t = transformUtils.transformFromPose(np.array([ 0.14376024,  0.95920689,  0.36655712]), np.array([ 0.28745842,  0.90741428, -0.28822068,  0.10438304]))
        self.leftFootEgressStartFrame  = vis.updateFrame(t, 'left foot start', scale=0.2,visible=True, parent=self.pointcloudAffordance)


        t = transformUtils.transformFromPose(np.array([ 0.26903719,  0.90783714,  0.24439189]),
                                             np.array([ 0.35290731,  0.93443693, -0.04181263,  0.02314636]))
        self.leftFootEgressMidFrame  = vis.updateFrame(t, 'left foot mid', scale=0.2,visible=True, parent=self.pointcloudAffordance)

        t = transformUtils.transformFromPose(np.array([ 0.54339115,  0.89436275,  0.26681047]),
                                             np.array([ 0.34635985,  0.93680077, -0.04152008,  0.02674412]))
        self.leftFootEgressOutsideFrame  = vis.updateFrame(t, 'left foot outside', scale=0.2,visible=True, parent=self.pointcloudAffordance)


        # pose = [np.array([-0.78962299,  0.44284877, -0.29539116]), np.array([ 0.54812954,  0.44571517, -0.46063251,  0.53731713])] #old location
        pose = [np.array([-0.78594663,  0.42026626, -0.23248139]), np.array([ 0.54812954,  0.44571517, -0.46063251,  0.53731713])] # updated location

        desc = dict(classname='CapsuleRingAffordanceItem', Name='Steering Wheel', uuid=newUUID(), pose=pose,
                    Color=[1, 0, 0], Radius=float(0.18), Segments=20)
        self.steeringWheelAffordance = segmentation.affordanceManager.newAffordanceFromDescription(desc)


        pose = [np.array([-0.05907324,  0.80460545,  0.45439687]), np.array([ 0.14288327,  0.685944  , -0.703969  ,  0.11615873])]

        desc = dict(classname='BoxAffordanceItem', Name='ground affordance', Dimensions=[0.12, 0.33, 0.04], pose=pose, Color=[0,1,0])
        self.pedalAffordance = segmentation.affordanceManager.newAffordanceFromDescription(desc)


        t = transformUtils.transformFromPose(np.array([ 0.04045136,  0.96565326,  0.25810111]),
            np.array([ 0.26484648,  0.88360091, -0.37065556, -0.10825996]))
        self.leftFootPedalSwingFrame = vis.updateFrame(t,'left foot pedal swing', scale=0.2, visible=True, parent=self.pointcloudAffordance)


        t = transformUtils.transformFromPose(np.array([-0.02562708,  0.91084703,  0.27375967]),
            np.array([ 0.10611078,  0.7280876 , -0.67537447,  0.04998264]))
        self.leftFootDrivingFrame = vis.updateFrame(t,'left foot driving', scale=0.2, visible=True, parent=self.pointcloudAffordance)

        t = transformUtils.transformFromPose(np.array([-0.12702725,  0.92068409,  0.27209386]), 
            np.array([ 0.2062255 ,  0.92155886, -0.30781119,  0.11598529]))
        self.leftFootDrivingKneeInFrame = vis.updateFrame(t,'left foot driving knee in', scale=0.2, visible=True, parent=self.pointcloudAffordance)

        t = transformUtils.transformFromPose(np.array([ 0.4720199 , -0.06517618,  0.00233972]), np.array([  6.10521653e-03,   4.18621358e-04,   4.65520611e-01,
                 8.85015882e-01]))
        self.rightHandGrabFrame = vis.updateFrame(t,'right hand grab bar', scale=0.2, visible=True, parent=self.pointcloudAffordance)

        self.frameSync = vis.FrameSync()
        self.frameSync.addFrame(self.originFrame)
        self.frameSync.addFrame(self.pointcloudAffordance.getChildFrame(), ignoreIncoming=True)
        self.frameSync.addFrame(self.leftFootEgressStartFrame, ignoreIncoming=True)
        self.frameSync.addFrame(self.leftFootEgressMidFrame, ignoreIncoming=True)
        self.frameSync.addFrame(self.leftFootEgressOutsideFrame, ignoreIncoming=True)
        self.frameSync.addFrame(self.steeringWheelAffordance.getChildFrame(), ignoreIncoming=True)
        self.frameSync.addFrame(self.pedalAffordance.getChildFrame(), ignoreIncoming=True)
        self.frameSync.addFrame(self.leftFootPedalSwingFrame, ignoreIncoming=True)
        self.frameSync.addFrame(self.leftFootDrivingFrame, ignoreIncoming=True)
        self.frameSync.addFrame(self.leftFootDrivingKneeInFrame, ignoreIncoming=True)
        self.frameSync.addFrame(self.rightHandGrabFrame, ignoreIncoming=True)

    def onAprilTag(self, msg):
        t = vtk.vtkTransform()
        cameraview.imageManager.queue.getTransform('april_tag_car_beam', 'local', msg.utime, t)
        self.originFrame.copyFrame(transformUtils.concatenateTransforms([self.originToAprilTransform, t]))

class EgressPlanner(object):

    def __init__(self, robotSystem):

        self.pelvisLiftX = 0.0
        self.pelvisLiftZ = 0.05

        self.legLiftAngle = 8

        self.coneThreshold = np.radians(5)

        self.robotSystem = robotSystem
        self.polaris = None
        self.quasiStaticShrinkFactor = 0.5
        self.maxBodyTranslationSpeed = 0.1
        self.plans = []

    def spawnPolaris(self):
        self.polaris = PolarisModel()

    def createLeftFootPoseConstraint(self, targetFrame, tspan=[-np.inf,np.inf]):
        positionConstraint, orientationConstraint = self.robotSystem.ikPlanner.createPositionOrientationConstraint('l_foot', targetFrame, vtk.vtkTransform())
        positionConstraint.tspan = tspan
        orientationConstraint.tspan = tspan
        return positionConstraint, orientationConstraint

    def createAllButLeftLegPostureConstraint(self, poseName):
        joints = robotstate.matchJoints('^(?!l_leg)')
        return self.robotSystem.ikPlanner.createPostureConstraint(poseName, joints)


    def getPlanningStartPose(self):
        return self.robotSystem.robotStateJointController.getPose('EST_ROBOT_STATE')

    def addPlan(self, plan):
        self.plans.append(plan)

    def commitManipPlan(self):
        self.robotSystem.manipPlanner.commitManipPlan(self.plans[-1])

    def planEgressArms(self):
        startPose = self.getPlanningStartPose()
        endPose = self.robotSystem.ikPlanner.getMergedPostureFromDatabase(startPose, 'driving', 'egress-arms')
        return self.robotSystem.ikPlanner.computePostureGoal(startPose, endPose)

    def planGetWeightOverFeet(self):
        startPose = self.getPlanningStartPose()
        startPoseName = 'q_egress_start'
        self.robotSystem.ikPlanner.addPose(startPose, startPoseName)
        endPoseName = 'q_egress_end'
        constraints = []
        constraints.append(ik.QuasiStaticConstraint(leftFootEnabled=True, rightFootEnabled=True,
                                                    pelvisEnabled=False,
                                                    shrinkFactor=self.quasiStaticShrinkFactor))
        constraints.append(self.robotSystem.ikPlanner.createLockedBasePostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createLockedLeftArmPostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createLockedRightArmPostureConstraint(startPoseName))
        constraintSet = ConstraintSet(self.robotSystem.ikPlanner, constraints, endPoseName, startPoseName)
        constraintSet.ikParameters = IkParameters(usePointwise=False, maxDegreesPerSecond=15)

        constraintSet.runIk()

        keyFramePlan =  constraintSet.planEndPoseGoal(feetOnGround=False)
        poseTimes, poses = planplayback.PlanPlayback.getPlanPoses(keyFramePlan)
        ts = [poseTimes[0]]
        supportsList = [['r_foot', 'l_foot', 'pelvis']]
        plan = self.publishPlanWithSupports(keyFramePlan, supportsList, ts, True)
        self.addPlan(plan)
        return plan

    def planStandUp(self):
        startPose = self.getPlanningStartPose()
        startPoseName = 'q_egress_start'
        self.robotSystem.ikPlanner.addPose(startPose, startPoseName)
        endPoseName = 'q_egress_end'
        pelvisFrame = self.robotSystem.ikPlanner.getLinkFrameAtPose('pelvis', startPose)
        t = transformUtils.frameFromPositionAndRPY([self.pelvisLiftX, 0, self.pelvisLiftZ], [0, 0, 0])
        liftFrame = transformUtils.concatenateTransforms([t, pelvisFrame])

        constraints = []
        p = ik.PositionConstraint(linkName='pelvis', referenceFrame=liftFrame,
                                  lowerBound=np.array([0.0, -np.inf, 0.0]),
                                  upperBound=np.array([np.inf, np.inf, 0.0]))
        constraints.append(p)
        constraints.append(ik.QuasiStaticConstraint(leftFootEnabled=True, rightFootEnabled=True, pelvisEnabled=False,
                                                    shrinkFactor=self.quasiStaticShrinkFactor))
        constraints.append(self.robotSystem.ikPlanner.createXYZMovingBasePostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createLockedLeftArmPostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createLockedRightArmPostureConstraint(startPoseName))
        constraints.extend(self.robotSystem.ikPlanner.createFixedFootConstraints(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createKneePostureConstraint([0.7, 2.5]))
        constraintSet = ConstraintSet(self.robotSystem.ikPlanner, constraints, endPoseName, startPoseName)
        constraintSet.ikParameters = IkParameters(usePointwise=True)

        constraintSet.runIk()
        keyFramePlan = constraintSet.planEndPoseGoal(feetOnGround=True)
        poseTimes, poses = planplayback.PlanPlayback.getPlanPoses(keyFramePlan)
        ts = [poseTimes[0]]
        supportsList = [['r_foot', 'l_foot']]
        plan = self.publishPlanWithSupports(keyFramePlan, supportsList, ts, True)
        self.addPlan(plan)
        return plan

    def createUtorsoGazeConstraints(self, tspan):
        constraints = []
        g = ik.WorldGazeDirConstraint()
        g.linkName = 'utorso'
        g.targetFrame = vtk.vtkTransform()
        axes = transformUtils.getAxesFromTransform(self.polaris.leftFootEgressOutsideFrame.transform)
        g.targetAxis = axes[0]
        g.bodyAxis = [1,0,0]
        g.coneThreshold = self.coneThreshold
        g.tspan = tspan
        constraints.append(g)

        g = ik.WorldGazeDirConstraint()
        g.linkName = 'utorso'
        g.targetFrame = vtk.vtkTransform()
        g.targetAxis = [0,0,1]
        g.bodyAxis = [0,0,1]
        g.coneThreshold = self.coneThreshold
        g.tspan = tspan
        constraints.append(g)
        return constraints

    def planShiftWeightOut(self):

        startPose = self.getPlanningStartPose()
        startPoseName = 'q_egress_start'
        self.robotSystem.ikPlanner.addPose(startPose, startPoseName)
        endPoseName = 'q_egress_end'
        constraints = []

        utorsoFrame = self.robotSystem.ikPlanner.getLinkFrameAtPose('utorso', startPose)
        constraints.extend(self.createUtorsoGazeConstraints([1.0, 1.0]))

        constraints.append(ik.QuasiStaticConstraint(leftFootEnabled=False, rightFootEnabled=True,
                                                    pelvisEnabled=False,
                                                    shrinkFactor=self.quasiStaticShrinkFactor))
        constraints.append(self.robotSystem.ikPlanner.createXYZMovingBasePostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createLockedLeftArmPostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createLockedRightArmPostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createFixedLinkConstraints(startPoseName, 'l_foot'))
        constraints.append(self.robotSystem.ikPlanner.createFixedLinkConstraints(startPoseName, 'r_foot'))
        constraintSet = ConstraintSet(self.robotSystem.ikPlanner, constraints, endPoseName, startPoseName)
        constraintSet.ikParameters = IkParameters(usePointwise=True)

        constraintSet.runIk()
        keyFramePlan = constraintSet.planEndPoseGoal(feetOnGround=False)
        poseTimes, poses = planplayback.PlanPlayback.getPlanPoses(keyFramePlan)
        ts = [poseTimes[0]]
        supportsList = [['r_foot', 'l_foot']]
        plan = self.publishPlanWithSupports(keyFramePlan, supportsList, ts, True)
        self.addPlan(plan)
        return plan

    def computeLeftFootOverPlatformFrame(self, startPose, height):
        lFoot2World = transformUtils.copyFrame(self.polaris.leftFootEgressOutsideFrame.transform)
        rFoot2World = self.robotSystem.ikPlanner.getLinkFrameAtPose('r_foot', startPose)
        lFoot2World.PostMultiply()
        lFoot2World.Translate(np.array(rFoot2World.GetPosition()) - lFoot2World.GetPosition())
        lFoot2World.PreMultiply()
        lFoot2World.Translate([0.05, 0.26, height])

        rFootRPY = transformUtils.rollPitchYawFromTransform(rFoot2World)
        lFootRPY = transformUtils.rollPitchYawFromTransform(lFoot2World);
        lFootxyz,_ = transformUtils.poseFromTransform(lFoot2World)

        lFootRPY[0] = rFootRPY[0]
        lFootRPY[1] = rFootRPY[1]
        lFoot2World = transformUtils.frameFromPositionAndRPY(lFootxyz, np.rad2deg(lFootRPY))
        return lFoot2World

    def planFootOut(self):

        startPose = self.getPlanningStartPose()
        startPoseName = 'q_egress_start'
        self.robotSystem.ikPlanner.addPose(startPose, startPoseName)
        endPoseName = 'q_egress_end'

        utorsoFrame = self.robotSystem.ikPlanner.getLinkFrameAtPose('utorso', startPose)
        finalLeftFootFrame = self.computeLeftFootOverPlatformFrame(startPose, 0.05)

        constraints = []
        constraints.extend(self.createUtorsoGazeConstraints([0.0, 1.0]))
        constraints.append(ik.QuasiStaticConstraint(leftFootEnabled=False, rightFootEnabled=True,
                                                    pelvisEnabled=False, shrinkFactor=0.01))
        constraints.append(self.robotSystem.ikPlanner.createMovingBaseSafeLimitsConstraint())
        constraints.append(self.robotSystem.ikPlanner.createLockedLeftArmPostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createLockedRightArmPostureConstraint(startPoseName))
        #constraints.append(self.robotSystem.ikPlanner.createLockedBackPostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createFixedLinkConstraints(startPoseName, 'r_foot'))
        constraints.extend(self.createLeftFootPoseConstraint(finalLeftFootFrame, tspan=[1,1]))

        constraintSet = ConstraintSet(self.robotSystem.ikPlanner, constraints, endPoseName, startPoseName)
        constraintSet.ikParameters = IkParameters(usePointwise=True, maxBaseRPYDegreesPerSecond=10,
                                                  rescaleBodyNames=['l_foot'],
                                                  rescaleBodyPts=[0.0, 0.0, 0.0],
                                                  maxBodyTranslationSpeed=self.maxBodyTranslationSpeed)
        #constraintSet.seedPoseName = 'q_start'
        #constraintSet.nominalPoseName = 'q_start'

        constraintSet.runIk()

        footFrame = self.robotSystem.ikPlanner.getLinkFrameAtPose('l_foot', startPose)
        t = transformUtils.frameFromPositionAndRPY([0, 0, self.polaris.leftFootEgressOutsideFrame.transform.GetPosition()[2]-footFrame.GetPosition()[2]], [0, 0, 0])
        liftFrame = transformUtils.concatenateTransforms([footFrame, t])
        vis.updateFrame(liftFrame, 'lift frame')

        c = ik.WorldFixedOrientConstraint()
        c.linkName = 'l_foot'
        c.tspan = [0.0, 0.1, 0.2]
        constraints.append(c)
        constraints.extend(self.createLeftFootPoseConstraint(liftFrame, tspan=[0.2, 0.2]))
        constraints.extend(self.createLeftFootPoseConstraint(self.polaris.leftFootEgressMidFrame, tspan=[0.5, 0.5]))

        constraints.extend(self.createLeftFootPoseConstraint(self.polaris.leftFootEgressOutsideFrame, tspan=[0.8, 0.8]))

        #plan = constraintSet.planEndPoseGoal(feetOnGround=False)
        keyFramePlan = constraintSet.runIkTraj()
        poseTimes, poses = planplayback.PlanPlayback.getPlanPoses(keyFramePlan)
        ts = [poseTimes[0]]
        supportsList = [['r_foot']]
        plan = self.publishPlanWithSupports(keyFramePlan, supportsList, ts, False)
        self.addPlan(plan)
        return plan

    def planLeftFootDown(self):
        startPose = self.getPlanningStartPose()
        startPoseName = 'q_footdown_start'
        self.robotSystem.ikPlanner.addPose(startPose, startPoseName)
        endPoseName = 'q_footdown_end'
        utorsoFrame = self.robotSystem.ikPlanner.getLinkFrameAtPose('utorso', startPose)
        finalLeftFootFrame = self.computeLeftFootOverPlatformFrame(startPose, 0.0)

        constraints = []
        constraints.extend(self.createUtorsoGazeConstraints([0.0, 1.0]))
        constraints.append(ik.QuasiStaticConstraint(leftFootEnabled=False, rightFootEnabled=True,
                                                    pelvisEnabled=False, shrinkFactor=0.01))
        constraints.append(self.robotSystem.ikPlanner.createMovingBaseSafeLimitsConstraint())
        constraints.append(self.robotSystem.ikPlanner.createLockedLeftArmPostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createLockedRightArmPostureConstraint(startPoseName))
        #constraints.append(self.robotSystem.ikPlanner.createLockedBackPostureConstraint(startPoseName))
        constraints.append(self.robotSystem.ikPlanner.createFixedLinkConstraints(startPoseName, 'r_foot'))
        constraints.extend(self.createLeftFootPoseConstraint(finalLeftFootFrame, tspan=[1,1]))

        constraintSet = ConstraintSet(self.robotSystem.ikPlanner, constraints, endPoseName, startPoseName)
        constraintSet.ikParameters = IkParameters(usePointwise=True)
        #constraintSet.seedPoseName = 'q_start'
        #constraintSet.nominalPoseName = 'q_start'

        constraintSet.runIk()
        keyFramePlan = constraintSet.planEndPoseGoal(feetOnGround=False)
        poseTimes, poses = planplayback.PlanPlayback.getPlanPoses(keyFramePlan)
        ts = [poseTimes[0], poseTimes[-1]]
        supportsList = [['r_foot'], ['r_foot','l_foot']]
        plan = self.publishPlanWithSupports(keyFramePlan, supportsList, ts, False)
        self.addPlan(plan)
        return plan


    def planCenterWeight(self):
        ikPlanner = self.robotSystem.ikPlanner
        startPose = self.getPlanningStartPose()
        startPoseName = 'q_lean_right'
        self.robotSystem.ikPlanner.addPose(startPose, startPoseName)
        endPoseName = 'q_egress_end'

        footFixedConstraints = ikPlanner.createFixedFootConstraints(startPoseName)
        backConstraint = ikPlanner.createMovingBackLimitedPostureConstraint()
        armsLocked = ikPlanner.createLockedArmsPostureConstraints(startPoseName)

        constraints = [backConstraint]
        constraints.extend(footFixedConstraints)
        constraints.extend(armsLocked)
        constraints.append(ik.QuasiStaticConstraint(leftFootEnabled=True, rightFootEnabled=True,
                                                    pelvisEnabled=False,
                                                    shrinkFactor=self.quasiStaticShrinkFactor))

        constraintSet = ConstraintSet(ikPlanner, constraints, endPoseName, startPoseName)
        constraintSet.seedPoseName = 'q_start'
        constraintSet.nominalPoseName = 'q_nom'
        endPose = constraintSet.runIk()
        keyFramePlan = constraintSet.planEndPoseGoal()
        poseTimes, poses = planplayback.PlanPlayback.getPlanPoses(keyFramePlan)
        ts = [poseTimes[0]]
        supportsList = [['r_foot','l_foot']]
        plan = self.publishPlanWithSupports(keyFramePlan, supportsList, ts, True)
        self.addPlan(plan)
        return plan


    def planArmsForward(self):
        q0 = self.getPlanningStartPose()
        q1 = self.robotSystem.ikPlanner.getMergedPostureFromDatabase(q0, 'General', 'hands-forward', side='left')
        q2 = self.robotSystem.ikPlanner.getMergedPostureFromDatabase(q1, 'General', 'hands-forward', side='right')
        q1 = 0.5*(q1 + np.array(q2))
        ikParameters = IkParameters(usePointwise=True, maxBaseRPYDegreesPerSecond=10,
                                    rescaleBodyNames=['l_hand', 'r_hand'],
                                    rescaleBodyPts=list(self.robotSystem.ikPlanner.getPalmPoint(side='left')) +
                                                    list(self.robotSystem.ikPlanner.getPalmPoint(side='right')),
                                    maxBodyTranslationSpeed=3*self.maxBodyTranslationSpeed)
        plan = self.robotSystem.ikPlanner.computeMultiPostureGoal([q0, q1, q2], ikParameters=ikParameters)
        self.addPlan(plan)
        return plan

    def publishPlanWithSupports(self, keyFramePlan, supportsList, ts, isQuasistatic):
        manipPlanner = self.robotSystem.manipPlanner
        msg_robot_plan_t = manipPlanner.convertKeyframePlan(keyFramePlan)
        supports = manipPlanner.getSupportLCMFromListOfSupports(supportsList,ts)
        msg_robot_plan_with_supports_t = manipPlanner.convertPlanToPlanWithSupports(msg_robot_plan_t, supports, ts, isQuasistatic)
        lcmUtils.publish('CANDIDATE_ROBOT_PLAN_WITH_SUPPORTS', msg_robot_plan_with_supports_t)
        return msg_robot_plan_with_supports_t

    def getFrameToOriginTransform(self, t):
        tCopy = transformUtils.copyFrame(t)
        tCopy.PostMultiply()
        tCopy.Concatenate(self.polaris.originFrame.transform.GetLinearInverse())
        print transformUtils.poseFromTransform(tCopy)
        return tCopy


class EgressPanel(TaskUserPanel):

    def __init__(self, robotSystem):

        TaskUserPanel.__init__(self, windowTitle='Egress')

        self.robotSystem = robotSystem
        self.egressPlanner = EgressPlanner(robotSystem)
        self.platformPlanner = polarisplatformplanner.PolarisPlatformPlanner(robotSystem.ikServer, robotSystem)
        self.addDefaultProperties()
        self.addButtons()
        self.addTasks()


    def addButtons(self):
        # Get onto platform buttons
        self.addManualButton('Spawn Polaris', self.egressPlanner.spawnPolaris)
        self.addManualButton('Get weight over feet', self.egressPlanner.planGetWeightOverFeet)
        self.addManualButton('Stand up', self.egressPlanner.planStandUp)
        self.addManualButton('Shift weight out', self.egressPlanner.planShiftWeightOut)
        self.addManualButton('Move left foot out', self.egressPlanner.planFootOut)
        self.addManualButton('Put foot down', self.egressPlanner.planLeftFootDown)
        self.addManualButton('Center weight', self.egressPlanner.planCenterWeight)
        self.addManualButton('Arms forward', self.egressPlanner.planArmsForward)
        self.addManualSpacer()
        #sit/stand buttons
        self.addManualButton('Start', self.onStart)
        # polaris step down buttons
        self.addManualButton('Fit Platform Affordance', self.platformPlanner.fitRunningBoardAtFeet)
        self.addManualButton('Spawn Ground Affordance', self.platformPlanner.spawnGroundAffordance)
        self.addManualButton('Raycast Terrain', self.platformPlanner.requestRaycastTerrain)
        self.addManualButton('Update Affordance', self.platformPlanner.updateAffordance)
        self.addManualButton('Arms Up',self.onArmsUp)
        self.addManualButton('Plan Step Down', self.onPlanStepDown)
        self.addManualButton('Plan Step Off', self.onPlanStepOff)

    def addDefaultProperties(self):
        self.params.addProperty('Step Off Direction', 0, attributes=om.PropertyAttributes(enumNames=['Forwards','Sideways']))

    def _syncProperties(self):
        self.stepOffDirection = self.params.getPropertyEnumValue('Step Off Direction').lower()

    def onStart(self):
        self._syncProperties()
        print 'Egress Planner Ready'

    def onUpdateAffordance(self):
        if not self.platformPlanner.initializedFlag:
            self.platformPlanner.initialize()

        self.platformPlanner.updateAffordance()

    def onPlan(self,planType):
        self._syncProperties()

    def onPlanTurn(self):
        self._syncProperties()
        self.platformPlanner.planTurn()

    def onArmsUp(self):
        self.platformPlanner.planArmsUp(self.stepOffDirection)

    def onPropertyChanged(self, propertySet, propertyName):
        self._syncProperties()

    def onPlanStepDown(self):
        self._syncProperties()
        if self.stepOffDirection == 'forwards':
            self.platformPlanner.planStepDownForwards()
        else:
            self.platformPlanner.planStepDown()

    def onPlanWeightShift(self):
        self._syncProperties()
        if self.stepOffDirection == 'forwards':
            self.platformPlanner.planWeightShiftForwards()
        else:
            self.platformPlanner.planWeightShift()

    def onPlanStepOff(self):
        self._syncProperties()
        if self.stepOffDirection == 'forwards':
            self.platformPlanner.planStepOffForwards()
        else:
            self.platformPlanner.planStepOff()

    def addTasks(self):



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

        def addManipTask(name, planFunc, userPrompt=False, planner=None):

            if planner is None:
                planner = self.platformPlanner
            prevFolder = self.folder
            addFolder(name, prevFolder)
            addFunc(planFunc, 'plan')
            if not userPrompt:
                addTask(rt.CheckPlanInfo(name='check manip plan info'))
            else:
                addTask(rt.UserPromptTask(name='approve manip plan', message='Please approve manipulation plan.'))
            addFunc(planner.commitManipPlan, name='execute manip plan')
            addTask(rt.WaitForManipulationPlanExecution(name='wait for manip execution'))
            self.folder = prevFolder

        pp = self.platformPlanner
        ep = self.egressPlanner

        stepOut = addFolder('Step out of car')
        self.folder = stepOut
        addManipTask('Get weight over feet', ep.planGetWeightOverFeet, userPrompt=True, planner=ep)
        addManipTask('Stand up', ep.planStandUp, userPrompt=True, planner=ep)
        addManipTask('Shift weight out of Polaris', ep.planShiftWeightOut, userPrompt=True, planner=ep)
        addManipTask('Move left foot out of Polaris', ep.planFootOut, userPrompt=True, planner=ep)
        addManipTask('Put left foot down on platform', ep.planLeftFootDown, userPrompt=True, planner=ep)
        addManipTask('Center weight over feet', ep.planCenterWeight, userPrompt=True, planner=ep)
        addManipTask('Move arms up for walking', ep.planArmsForward, userPrompt=True, planner=ep)

        prep = addFolder('Step down prep')
        addFunc(self.onStart, 'start')
        addTask(rt.SetNeckPitch(name='set neck position', angle=60))
        # addManipTask('arms up', self.onArmsUp, userPrompt=True)
        addTask(rt.UserPromptTask(name="confirm arms up", message="Please confirm arms up, if not use 'Arms Up' Button"))
        self.folder = prep
        addFunc(pp.fitRunningBoardAtFeet, 'fit running board')
        addFunc(pp.spawnGroundAffordance, 'spawn ground affordance')
        addFunc(pp.requestRaycastTerrain, 'raycast terrain')
        addTask(rt.UserPromptTask(name="set walking params", message="Please set walking params to 'Polaris Platform'"))

        folder = addFolder('Step Down')
        addFunc(pp.spawnGroundAffordance, 'spawn ground affordance')
        addFunc(pp.requestRaycastTerrain, 'raycast terrain')
        addFunc(self.onPlanStepDown, 'plan step down')
        addTask(rt.UserPromptTask(name="approve footsteps, set support contact group",
         message="Please approve/modify footsteps. Set the support contact group for the left foot step to be Back 2/3"))
        addFunc(self.robotSystem.footstepsDriver.onExecClicked, 'commit footstep plan')
        addTask(rt.WaitForWalkExecution(name='wait for walking'))

        folder = addFolder('Step Off')
        addFunc(pp.spawnGroundAffordance, 'spawn ground affordance')
        addFunc(pp.requestRaycastTerrain, 'raycast terrain')
        addFunc(self.onPlanStepOff, 'plan step off')
        addTask(rt.UserPromptTask(name="approve footsteps", message="Please approve footsteps, modify if necessary"))
        addFunc(self.robotSystem.footstepsDriver.onExecClicked, 'commit footstep plan')
        addTask(rt.WaitForWalkExecution(name='wait for walking'))
        addManipTask('plan nominal', pp.planNominal, userPrompt=True)

