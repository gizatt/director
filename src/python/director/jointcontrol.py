import os
import math
from director.timercallback import TimerCallback
from director.simpletimer import SimpleTimer
from director import robotstate
from director import getDRCBaseDir
from director import transformUtils
from director import lcmUtils
import drc as lcmdrc
import bot_core
import numpy as np


class JointController(object):

    def __init__(self, models, poseCollection=None, jointNames=None):
        self.jointNames = jointNames or robotstate.getDrakePoseJointNames()
        self.jointMap = None
        self.numberOfJoints = len(self.jointNames)
        self.models = list(models)
        self.poses = {}
        self.poseCollection = poseCollection
        self.currentPoseName = None
        self.lastRobotStateMessage = None
        self.ignoreOldStateMessages = False
        self.setPose('q_zero', np.zeros(self.numberOfJoints))

    def setJointPosition(self, jointId, position):
        '''
        Set joint position in degrees.
        '''
        assert jointId >= 0 and jointId < len(self.q)
        self.q[jointId] = math.radians(position % 360.0)
        self.push()

    def push(self):
        for model in self.models:
            model.model.setJointPositions(self.q, self.jointNames)

    def setPose(self, poseName, poseData=None, pushToModel=True):
        if poseData is not None:
            self.addPose(poseName, poseData)
        if poseName not in self.poses:
            raise Exception('Pose %r has not been defined.' % poseName)
        self.q = self.poses[poseName]
        self.currentPoseName = poseName
        if pushToModel:
            self.push()

    def setZeroPose(self):
        self.setPose('q_zero')

    def getPose(self, poseName):
        return self.poses.get(poseName)

    def addPose(self, poseName, poseData):
        assert len(poseData) == self.numberOfJoints
        self.poses[poseName] = np.asarray(poseData)
        if self.poseCollection is not None:
            self.poseCollection.setItem(poseName, poseData)

    def loadPoseFromFile(self, filename):
        assert os.path.splitext(filename)[1] == '.mat'
        import scipy.io
        matData = scipy.io.loadmat(filename)
        return np.array(matData['xstar'][:self.numberOfJoints].flatten(), dtype=float)

    def regenerateJointMap(self, msgJointNames):
        self.jointMap = dict()
        for name in msgJointNames:
            if name in self.jointNames:
                self.jointMap[name] = self.jointNames.index(name) 
            else:
                self.jointMap[name] = -1


    def addLCMUpdater(self, channelName):
        '''
        adds an lcm subscriber to update the joint positions from
        lcm robot_state_t messages
        '''

        def onRobotStateMessage(msg):
            if self.ignoreOldStateMessages and self.lastRobotStateMessage is not None and msg.utime < self.lastRobotStateMessage.utime:
                return
            poseName = channelName

            # map from msg joint names to our drake joint names
            if self.jointMap is None:
                self.regenerateJointMap(msg.joint_name)

            pose = np.zeros(self.numberOfJoints)
            trans = msg.pose.translation
            quat = msg.pose.rotation
            pose[0:3] = [trans.x, trans.y, trans.z]
            quat = [quat.w, quat.x, quat.y, quat.z]
            pose[3:6] = transformUtils.quaternionToRollPitchYaw(quat)
            for name, position in zip(msg.joint_name, msg.joint_position):
                if self.jointMap[name] >= 0:
                    pose[self.jointMap[name]] = position

            self.lastRobotStateMessage = msg

            # use joint name/positions from robot_state_t and append base_{x,y,z,roll,pitch,yaw}
            jointPositions = np.hstack((msg.joint_position, pose[:6]))
            jointNames = msg.joint_name + robotstate.getDrakePoseJointNames()[:6]

            self.setPose(poseName, pose, pushToModel=False)
            for model in self.models:
                model.model.setJointPositions(jointPositions, jointNames)

        self.subscriber = lcmUtils.addSubscriber(channelName, bot_core.robot_state_t, onRobotStateMessage)
        self.subscriber.setSpeedLimit(60)

    def removeLCMUpdater(self):
        lcmUtils.removeSubscriber(self.subscriber)
        self.subscriber = None


class JointControlTestRamp(TimerCallback):

    def __init__(self, jointController):
        TimerCallback.__init__(self)
        self.controller = jointController
        self.testTime = 2.0

    def testJoint(self, jointId):
        self.jointId = jointId
        self.testTimer = SimpleTimer()
        self.start()

    def tick(self):

        if self.testTimer.elapsed() > self.testTime:
            self.stop()
            return

        jointPosition = math.sin( (self.testTimer.elapsed() / self.testTime) * math.pi) * math.pi
        self.controller.setJointPosition(self.jointId, math.degrees(jointPosition))
