
from director.consoleapp import ConsoleApp
from director import robotsystem
from director import transformUtils
from director import vtkAll as vtk
from director.utime import getUtime
from director import lcmUtils
from director.timercallback import TimerCallback
from director import roboturdf

import numpy as np
import math

import drc as lcmdrc
import bot_core as lcmbot

#---------------------------------------

app = ConsoleApp()
app.setupGlobals(globals())
app.showPythonConsole()
view = app.createView()
view.show()
roboturdf.loadRobotModel('urdf model', view, urdfFile=sys.argv[1])

app.start()
