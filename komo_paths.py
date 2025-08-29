import robotic as ry
import numpy as np
import pprint

def _komo_wrapper(komo_path):
    def wrapper(C: ry.Config, *args, q_start=None, verbose=False, show_result=False, no_wrap=False, **kwargs):
        if no_wrap:
            return komo_path(C, *args, **kwargs)
        if q_start is not None:
            q_current = C.getJointState()
            C.setJointState(q_start)

        komo = komo_path(C, *args, **kwargs)
        solver = ry.NLP_Solver(komo.nlp(), verbose=-1)
        ret = solver.solve()

        if verbose:
            print(f"feasible: {ret.feasible}")
            if not ret.feasible:
                pprint.pp(komo.report(), depth=2)
                pprint.pp(solver.reportLagrangeGradients(komo.getFeatureNames()), depth=2)
        if show_result:
            komo.view(True)

        if q_start is not None:
            C.setJointState(q_current)
        
        return komo.getPath(), ret.feasible
    return wrapper


@_komo_wrapper
def search_path(C):
    komo = ry.KOMO(C, 6, 10, 2, True)
    
    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 2, 1e0)

    # Feasibility Constraints
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    # search path consists of 2 circles at constant height
    komo.addObjective([1,6], ry.FS.position, ['l_gripper'], ry.OT.eq, [0,0,1], [0, 0, 1])
    # big circle
    komo.addObjective([1,3], ry.FS.negDistance, ['l_gripper', 'l_panda_base'], ry.OT.eq, [1], [-.7])
    #komo.addObjective([1], ry.FS.jointState, [], ry.OT.eq, order=1)
    # small circle
    komo.addObjective([4,6], ry.FS.negDistance, ['l_gripper', 'l_panda_base'], ry.OT.eq, [1], [-0.5])

    # to make sure we move in a half circle we want to be left of the robot at the start
    # then in front of it, the on its right
    # big circle
    
    komo.addObjective([1], ry.FS.positionDiff, ['l_gripper', 'l_panda_base'], ry.OT.eq, [0,1,0], [0])
    komo.addObjective([1], ry.FS.positionDiff, ['l_panda_base', 'l_gripper'], ry.OT.ineq, [1,0,0], [0])

    komo.addObjective([2], ry.FS.positionDiff, ['l_gripper', 'l_panda_base'], ry.OT.eq, [1,0,0], [0])
    komo.addObjective([2], ry.FS.positionDiff, ['l_panda_base', 'l_gripper'], ry.OT.ineq, [0,1,0], [0])
    
    komo.addObjective([3], ry.FS.positionDiff, ['l_gripper', 'l_panda_base'], ry.OT.eq, [0,1,0], [0])
    komo.addObjective([3], ry.FS.positionDiff, ['l_panda_base', 'l_gripper'], ry.OT.ineq, [-1,0,0], [0])

    
    # small circle
    komo.addObjective([4], ry.FS.positionDiff, ['l_gripper', 'l_panda_base'], ry.OT.eq, [0,1,0], [0])
    komo.addObjective([4], ry.FS.positionDiff, ['l_panda_base', 'l_gripper'], ry.OT.ineq, [-1,0,0], [0])

    komo.addObjective([5], ry.FS.positionDiff, ['l_gripper', 'l_panda_base'], ry.OT.eq, [1,0,0], [0])
    komo.addObjective([5], ry.FS.positionDiff, ['l_panda_base', 'l_gripper'], ry.OT.ineq, [0,1,0], [0])

    komo.addObjective([6], ry.FS.positionDiff, ['l_gripper', 'l_panda_base'], ry.OT.eq, [0,1,0], [0])
    komo.addObjective([6], ry.FS.positionDiff, ['l_panda_base', 'l_gripper'], ry.OT.ineq, [1,0,0], [0])

    # point camera downwards
    komo.addObjective([1,6], ry.FS.scalarProductXZ, ['world','l_gripper'], ry.OT.eq, [1])
    komo.addObjective([1,6], ry.FS.scalarProductYZ, ['world','l_gripper'], ry.OT.eq, [1])

    return komo



@_komo_wrapper
def search_path_orient_gripper(C):
    komo = search_path(C, no_wrap=True)
    komo.addObjective([1,3], ry.FS.scalarProductXY, ['world','l_gripper'], ry.OT.eq, [1])
    komo.addObjective([4,6], ry.FS.scalarProductXY, ['world','l_gripper'], ry.OT.eq, [1])
    return komo


@_komo_wrapper
def look_at_target(C, target, height):
    q0 = C.getJointState()
    komo = ry.KOMO(C, 1, 1, 0, True)

    above_target = target + np.array([0,0, height])
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q0) #cost: close to 'current state'

    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1])
    komo.addObjective([], ry.FS.position, ['l_cameraWrist'], ry.OT.eq, np.eye(3), above_target) 
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    # point camera downwards
    komo.addObjective([], ry.FS.scalarProductXZ, ['world','l_gripper'], ry.OT.eq, [1])
    komo.addObjective([], ry.FS.scalarProductYZ, ['world','l_gripper'], ry.OT.eq, [1])
    # align gripper coords with global coords, used for calibration
    komo.addObjective([], ry.FS.scalarProductXY, ['world','l_gripper'], ry.OT.eq, [1])
    komo.addObjective([],ry.FS.scalarProductXX, ['world','l_gripper'], ry.OT.ineq, [-1])
    return komo


@_komo_wrapper
def move_down(C, amount):
    q0 = C.getJointState()
    position, _ = C.eval(ry.FS.position, ['l_gripper'])
    komo = ry.KOMO(C, 1, 1, 0, True)

    target = position - np.array([0,0, amount])
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q0) #cost: close to 'current state'

    komo.addObjective([], ry.FS.position, ['l_gripper'], ry.OT.eq, [1e1], target) 
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    return komo




@_komo_wrapper
def touch_point(C, target_name):
    q0 = C.getJointState()
    komo = ry.KOMO(C, 1, 1, 0, False) #one phase one time slice problem, with 'delta_t=1', order=0
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q0) #cost: close to 'current state'
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1])
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.positionDiff, ['l_gripper', target_name], ry.OT.eq, [1], [0,0,0.01]) #constraint: gripper position

    komo.addObjective([], ry.FS.scalarProductXZ, ['world','l_gripper'], ry.OT.eq, [1], [0]) 
    komo.addObjective([], ry.FS.scalarProductYZ, ['world','l_gripper'], ry.OT.eq, [1], [0])

    return komo

@_komo_wrapper
def approach_straight(C, target_name, min_dist_final_approach=.2, safety_dist=.01, z_overwrite=None):
    komo = ry.KOMO(C, 2, 10, 2, True)
    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 2, 1e0)

    # Feasibility Constraints
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    # position
    # be above goal
    komo.addObjective([0,1], ry.FS.positionDiff, [target_name,'l_gripper'], ry.OT.ineq, [0,0,1], [0,0,-min_dist_final_approach]) 
    # arrive at goal 
    if z_overwrite is None:
        komo.addObjective([2], ry.FS.positionDiff, ['l_gripper', target_name], ry.OT.eq, [1], [0,0,safety_dist]) 
    else:
        komo.addObjective([2], ry.FS.positionDiff, ['l_gripper', target_name], ry.OT.eq, [[1,0,0],[0,1,0]], [0]) 
        # we need to scale the next constraint to make sure the EE ends up close to the table
        komo.addObjective([2], ry.FS.position, ['l_gripper'], ry.OT.eq, [0,0,1e1], [0,0,z_overwrite + safety_dist])
    # keep xy aligned in final approach
    komo.addObjective([1,2], ry.FS.positionDiff, [target_name,'l_gripper'], ry.OT.eq, [[1, 0, 0],[0, 1, 0]]) 

    # orientation
    # orient gripper to point downwards before last approach
    komo.addObjective([1,2], ry.FS.scalarProductXZ, ['world','l_gripper'], ry.OT.eq, [1], [0]) 
    komo.addObjective([1,2], ry.FS.scalarProductYZ, ['world','l_gripper'], ry.OT.eq, [1], [0])
    # speed
    # komo.addObjective([1], ry.FS.jointState, [], ry.OT.eq, [1], order=1)
    komo.addObjective([2], ry.FS.jointState, [], ry.OT.eq, [1], order=1)

    return komo


@_komo_wrapper
def connect_points(C: ry.Config, markers: list[str], safety_dist=.01, z_overwrite=None):
    komo = ry.KOMO(C, len(markers), 10, 1, True)
    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1)

    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)

    #keep in contact with table and pointed downwards
    if z_overwrite is None:
        komo.addObjective([], ry.FS.positionDiff, ["l_gripper", markers[0]], ry.OT.eq, [0,0,1], [safety_dist])
    else:
        komo.addObjective([], ry.FS.position, ["l_gripper"], ry.OT.eq, [0,0,1e1], [z_overwrite + safety_dist])
    komo.addObjective([], ry.FS.scalarProductXZ, ["table", "l_gripper"], ry.OT.eq)
    komo.addObjective([], ry.FS.scalarProductYZ, ["table", "l_gripper"], ry.OT.eq)

    # visit markers
    for i, (marker, next_marker) in enumerate(zip(markers[:-1], markers[1:])):
        komo.addObjective([i+1], ry.FS.positionDiff, ["l_gripper", marker], ry.OT.eq, [1,1,0])
        komo.addObjective([i+1], ry.FS.jointState, [], ry.OT.eq, order=1)
        position = C.getFrame(marker).getPosition()
        next_position = C.getFrame(next_marker).getPosition()
        line_vec = next_position[:2] - position[:2]  # we care only about the xy plane
        projection = np.outer(line_vec, line_vec)/np.inner(line_vec, line_vec)
        orthogonal = np.eye(2) - projection
        target = position
        scale = np.hstack([orthogonal, np.zeros((2,1))])
        komo.addObjective([i+1,i+2], ry.FS.position, ["l_gripper"], ry.OT.eq, scale, target)
            
    komo.addObjective([len(markers)], ry.FS.positionDiff, ["l_gripper", markers[-1]], ry.OT.eq, [1,1,0])
    komo.addObjective([len(markers)], ry.FS.jointState, [], ry.OT.eq, order=1) # always break at last marker

    return komo