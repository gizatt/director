import PythonQt
from PythonQt import QtCore, QtGui
import director.objectmodel as om
import director.visualization as vis
from director import cameracontrol
from director import propertyset
from director import frameupdater
from director import vieweventfilter


_contextMenuActions = []

def registerContextMenuActions(getActionsFunction):
    _contextMenuActions.append(getActionsFunction)


def getContextMenuActions(view, pickedObj, pickedPoint):
    actions = []
    for func in _contextMenuActions:
        actions.extend(func(view, pickedObj, pickedPoint))
    return actions

def getDefaultContextMenuActions(view, pickedObj, pickedPoint):

    def onDelete():
        om.removeFromObjectModel(pickedObj)

    def onHide():
        pickedObj.setProperty('Visible', False)

    def onSelect():
        om.setActiveObject(pickedObj)

    actions = [
      (None, None),
      ('Select', onSelect),
      ('Hide', onHide)
      ]

    if pickedObj.getProperty('Deletable'):
        actions.append(['Delete', onDelete])

    return actions

registerContextMenuActions(getDefaultContextMenuActions)


def getShortenedName(name, maxLength=30):
    if len(name) > maxLength:
        name = name[:maxLength-3] + '...'
    return name


def showRightClickMenu(displayPoint, view):

    pickedObj, pickedPoint = vis.findPickedObject(displayPoint, view)
    if not pickedObj:
        return

    objectName = pickedObj.getProperty('Name')
    if objectName == 'grid':
        return

    objectName = getShortenedName(objectName)

    displayPoint = displayPoint[0], view.height - displayPoint[1]

    globalPos = view.mapToGlobal(QtCore.QPoint(*displayPoint))

    menu = QtGui.QMenu(view)

    widgetAction = QtGui.QWidgetAction(menu)
    label = QtGui.QLabel('<b>%s</b>' % objectName)
    label.setContentsMargins(9,9,6,6)
    widgetAction.setDefaultWidget(label)
    menu.addAction(widgetAction)
    menu.addSeparator()


    propertiesPanel = PythonQt.dd.ddPropertiesPanel()
    propertiesPanel.setBrowserModeToWidget()
    propertyset.PropertyPanelHelper.addPropertiesToPanel(pickedObj.properties, propertiesPanel)

    def onPropertyChanged(prop):
        om.PropertyPanelHelper.setPropertyFromPanel(prop, propertiesPanel, pickedObj.properties)
    propertiesPanel.connect('propertyValueChanged(QtVariantProperty*)', onPropertyChanged)

    propertiesMenu = menu.addMenu('Properties')
    propertiesWidgetAction = QtGui.QWidgetAction(propertiesMenu)
    propertiesWidgetAction.setDefaultWidget(propertiesPanel)
    propertiesMenu.addAction(propertiesWidgetAction)

    def onDelete():
        om.removeFromObjectModel(pickedObj)

    def onHide():
        pickedObj.setProperty('Visible', False)

    def onSelect():
        om.setActiveObject(pickedObj)

    reachFrame = getAsFrame(pickedObj)
    collisionParent = getCollisionParent(pickedObj)
    def onReachLeft():
        reachToFrame(reachFrame, 'left', collisionParent)
    def onReachRight():
        reachToFrame(reachFrame, 'right', collisionParent)

    def flipHandSide():
        for obj in [pickedObj] + pickedObj.children():
            if not isGraspSeed(obj):
                continue
            side = 'right' if obj.side == 'left' else 'left'
            obj.side = side
            color = [1.0, 1.0, 0.0]
            if side == 'right':
                color = [0.33, 1.0, 0.0]
            obj.setProperty('Color', color)

            polyData = handFactory.getNewHandPolyData(side)
            obj.setPolyData(polyData)

            handFrame = obj.children()[0]
            t = transformUtils.copyFrame(handFrame.transform)
            t.PreMultiply()
            t.RotateY(180)
            handFrame.copyFrame(t)

            objName = obj.getProperty('Name')
            frameName = handFrame.getProperty('Name')
            if side == 'left':
                obj.setProperty('Name', objName.replace("right", "left"))
                handFrame.setProperty('Name', frameName.replace("right", "left"))
            else:
                obj.setProperty('Name', objName.replace("left", "right"))
                handFrame.setProperty('Name', frameName.replace("left", "right"))
            obj._renderAllViews()

    def flipHandThumb():
        handFrame = pickedObj.children()[0]
        t = transformUtils.copyFrame(handFrame.transform)
        t.PreMultiply()
        t.RotateY(180)
        handFrame.copyFrame(t)
        pickedObj._renderAllViews()

    def onSplineLeft():
        splinewidget.planner.newSpline(pickedObj, 'left')
    def onSplineRight():
        splinewidget.planner.newSpline(pickedObj, 'right')


    def getPointCloud(obj):
        try:
            obj = obj.model.polyDataObj
        except AttributeError:
            pass
        try:
            obj.polyData
        except AttributeError:
            return None
        if obj and obj.polyData.GetNumberOfPoints():# and (obj.polyData.GetNumberOfCells() == obj.polyData.GetNumberOfVerts()):
            return obj


    pointCloudObj = getPointCloud(pickedObj)
    affordanceObj = pickedObj if isinstance(pickedObj, affordanceitems.AffordanceItem) else None

    def onSegmentGround():
        groundPoints, scenePoints =  segmentation.removeGround(pointCloudObj.polyData)
        vis.showPolyData(groundPoints, 'ground points', color=[0,1,0], parent='segmentation')
        vis.showPolyData(scenePoints, 'scene points', color=[1,0,1], parent='segmentation')
        pickedObj.setProperty('Visible', False)


    def onCopyPointCloud():
        global lastRandomColor
        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(pointCloudObj.polyData)
        
        if pointCloudObj.getChildFrame():
            polyData = segmentation.transformPolyData(polyData, pointCloudObj.getChildFrame().transform)
        polyData = segmentation.addCoordArraysToPolyData(polyData)

        # generate random color, and average with a common color to make them generally similar
        lastRandomColor = lastRandomColor + 0.1 + 0.1*random.random()
        rgb = colorsys.hls_to_rgb(lastRandomColor, 0.7, 1.0)
        obj = vis.showPolyData(polyData, pointCloudObj.getProperty('Name') + ' copy', color=rgb, parent='point clouds')

        #t = vtk.vtkTransform()
        #t.PostMultiply()
        #t.Translate(filterUtils.computeCentroid(polyData))
        #segmentation.makeMovable(obj, t)
        om.setActiveObject(obj)
        pickedObj.setProperty('Visible', False)

    def onMergeIntoPointCloud():
        allPointClouds = om.findObjectByName('point clouds')
        if allPointClouds:
            allPointClouds = [i.getProperty('Name') for i in allPointClouds.children()]
        sel =  QtGui.QInputDialog.getItem(None, "Point Cloud Merging", "Pick point cloud to merge into:", allPointClouds, current=0, editable=False)
        sel = om.findObjectByName(sel)

        # Make a copy of each in same frame
        polyDataInto = vtk.vtkPolyData()
        polyDataInto.ShallowCopy(sel.polyData)
        if sel.getChildFrame():
            polyDataInto = segmentation.transformPolyData(polyDataInto, sel.getChildFrame().transform)

        polyDataFrom = vtk.vtkPolyData()
        polyDataFrom.DeepCopy(pointCloudObj.polyData)
        if pointCloudObj.getChildFrame():
            polyDataFrom = segmentation.transformPolyData(polyDataFrom, pointCloudObj.getChildFrame().transform)

        # Actual merge
        append = filterUtils.appendPolyData([polyDataFrom, polyDataInto])
        if sel.getChildFrame():
            polyDataInto = segmentation.transformPolyData(polyDataInto, sel.getChildFrame().transform.GetInverse())

        # resample
        append = segmentationroutines.applyVoxelGrid(append, 0.01)
        append = segmentation.addCoordArraysToPolyData(append)

        # Recenter the frame
        sel.setPolyData(append)
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(filterUtils.computeCentroid(append))
        segmentation.makeMovable(sel, t)

        # Hide the old one
        if pointCloudObj.getProperty('Name') in allPointClouds:
            pointCloudObj.setProperty('Visible', False)


    def onSegmentTableScene():
        polyData = pointCloudObj.polyData
        if pointCloudObj.actor.GetUserTransform():
            polyData = filterUtils.transformPolyData(polyData, pointCloudObj.actor.GetUserTransform())
        data = segmentation.segmentTableScene(polyData, pickedPoint)
        vis.showClusterObjects(data.clusters + [data.table], parent='segmentation')

    def onSegmentDrillAlignedWithTable():
        segmentation.segmentDrillAlignedWithTable(pickedPoint, pointCloudObj.polyData)

    def onCachePickedPoint():
        ''' Cache the Picked Point for general purpose use'''
        global lastCachedPickedPoint
        lastCachedPickedPoint = pickedPoint
        #data = segmentation.segmentTableScene(pointCloudObj.polyData, pickedPoint)
        #vis.showClusterObjects(data.clusters + [data.table], parent='segmentation')


    def onLocalPlaneFit():
        planePoints, normal = segmentation.applyLocalPlaneFit(pointCloudObj.polyData, pickedPoint, searchRadius=0.1, searchRadiusEnd=0.2)
        obj = vis.showPolyData(planePoints, 'local plane fit', color=[0,1,0])
        obj.setProperty('Point Size', 7)

        fields = segmentation.makePolyDataFields(obj.polyData)

        pose = transformUtils.poseFromTransform(fields.frame)
        desc = dict(classname='BoxAffordanceItem', Name='local plane', Dimensions=list(fields.dims), pose=pose)
        box = segmentation.affordanceManager.newAffordanceFromDescription(desc)

    def onOrientToMajorPlane():
        polyData, planeFrame = segmentation.orientToMajorPlane(pointCloudObj.polyData, pickedPoint=pickedPoint)
        pointCloudObj.setPolyData(polyData)


    def onDiskGlyph():
        result = segmentation.applyDiskGlyphs(pointCloudObj.polyData)
        obj = vis.showPolyData(result, 'disks', color=[0.8,0.8,0.8])
        om.setActiveObject(obj)
        pickedObj.setProperty('Visible', False)

    def onArrowGlyph():
        result = segmentation.applyArrowGlyphs(pointCloudObj.polyData)
        obj = vis.showPolyData(result, 'disks')

    def onSegmentationEditor():
        segmentationpanel.activateSegmentationMode(pointCloudObj.polyData)

    def addNewFrame():
        t = transformUtils.copyFrame(affordanceObj.getChildFrame().transform)
        t.PostMultiply()
        t.Translate(np.array(pickedPoint) - np.array(t.GetPosition()))
        newFrame = vis.showFrame(t, '%s frame %d' % (affordanceObj.getProperty('Name'), len(affordanceObj.children())), scale=0.2, parent=affordanceObj)
        affordanceObj.getChildFrame().getFrameSync().addFrame(newFrame, ignoreIncoming=True)

    def copyAffordance():
        desc = dict(affordanceObj.getDescription())
        del desc['uuid']
        desc['Name'] = desc['Name'] + ' copy'
        aff = robotSystem.affordanceManager.newAffordanceFromDescription(desc)
        aff.getChildFrame().setProperty('Edit', True)

    def onPromoteToAffordance():
        affObj = affordanceitems.MeshAffordanceItem.promotePolyDataItem(pickedObj)
        robotSystem.affordanceManager.registerAffordance(affObj)

    actions = [
      (None, None),
      ('Hide', onHide),
      ('Delete', onDelete),
      ('Select', onSelect)
      ]


    if affordanceObj:
        actions.extend([
            ('Copy affordance', copyAffordance),
            ('Add new frame', addNewFrame),
        ])

    elif type(pickedObj) == vis.PolyDataItem:
        actions.extend([
            ('Promote to Affordance', onPromoteToAffordance),
        ])

    if isGraspSeed(pickedObj):
        actions.extend([
            (None, None),
            ('Flip Side', flipHandSide),
            ('Flip Thumb', flipHandThumb),
        ])

    if reachFrame is not None:
        actions.extend([
            (None, None),
            ('Reach Left', onReachLeft),
            ('Reach Right', onReachRight),
            #('Spline Left', onSplineLeft),
            #('Spline Right', onSplineRight),
            ])

    if pointCloudObj:
        actions.extend([
            (None, None),
            ('Copy Pointcloud', onCopyPointCloud),
            ('Merge Pointcloud Into', onMergeIntoPointCloud),
            ('Segment Ground', onSegmentGround),
            ('Segment Table', onSegmentTableScene),
            ('Segment Drill Aligned', onSegmentDrillAlignedWithTable),
            ('Local Plane Fit', onLocalPlaneFit),
            ('Orient with Horizontal', onOrientToMajorPlane),
            ('Arrow Glyph', onArrowGlyph),
            ('Disk Glyph', onDiskGlyph),
            ('Cache Pick Point', onCachePickedPoint),
            (None, None),
            ('Open Segmentation Editor', onSegmentationEditor)
            ])

    for actionName, func in actions:
        if not actionName:
            menu.addSeparator()
        else:
            action = menu.addAction(actionName)
            action.connect('triggered()', func)

    selectedAction = menu.popup(globalPos)


def zoomToPick(displayPoint, view):
    pickedPoint, prop, _ = vis.pickProp(displayPoint, view)
    if not prop:
        return
    flyer = cameracontrol.Flyer(view)
    flyer.zoomTo(pickedPoint)


def getChildFrame(obj):
    if hasattr(obj, 'getChildFrame'):
        return obj.getChildFrame()


def toggleFrameWidget(displayPoint, view):

    obj, _ = vis.findPickedObject(displayPoint, view)

    if not isinstance(obj, vis.FrameItem):
        obj = getChildFrame(obj)

    if not obj:
        return False

    edit = not obj.getProperty('Edit')
    obj.setProperty('Edit', edit)

    parent = obj.parent()
    if getChildFrame(parent) == obj:
        parent.setProperty('Alpha', 0.5 if edit else 1.0)

    return True


class ViewBehaviors(vieweventfilter.ViewEventFilter):


    def onLeftDoubleClick(self, event):

        displayPoint = vis.mapMousePosition(self.view, event)
        if toggleFrameWidget(displayPoint, self.view):
            self.consumeEvent()

    def onRightClick(self, event):
        displayPoint = vis.mapMousePosition(self.view, event)
        showRightClickMenu(displayPoint, self.view)


    def onKeyPress(self, event):

        consumed = False

        key = str(event.text()).lower()

        if key == 'f':
            consumed = True
            zoomToPick(self.getCursorDisplayPosition(), self.view)

        elif key == 'r':
            consumed = True
            self.view.resetCamera()
            self.view.render()

        if consumed:
            self.consumeEvent()

    def onKeyPressRepeat(self, event):

        consumed = frameupdater.handleKey(event)

        # prevent these keys from going to vtkRenderWindow's default key press handler
        key = str(event.text()).lower()
        if key in ['r', 's', 'w', 'l', '3']:
            consumed = True

        if consumed:
            self.consumeEvent()
