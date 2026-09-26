"""Paste into the Isaac Sim Script Editor to publish the robot's true pose.

Adds an OmniGraph that publishes the world pose of the robot chassis on
/ground_truth_tf as a TF message (parent 'World', child 'chassis_link').
It uses its own topic so it never collides with the Nav2 TF tree.
Save the stage afterwards so the graph persists.
"""

import omni.graph.core as og
import omni.usd
import usdrt.Sdf

GRAPH = "/World/GroundTruthGraph"
ROBOT_PRIM = "/World/Nova_Carter_ROS/chassis_link"

stage = omni.usd.get_context().get_stage()
if stage.GetPrimAtPath(GRAPH):
    stage.RemovePrim(GRAPH)
if not stage.GetPrimAtPath(ROBOT_PRIM):
    raise RuntimeError(f"robot prim not found: {ROBOT_PRIM}")

keys = og.Controller.Keys
og.Controller.edit(
    {"graph_path": GRAPH, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("context", "isaacsim.ros2.bridge.ROS2Context"),
            ("sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("pub_tf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
        ],
        keys.CONNECT: [
            ("tick.outputs:tick", "pub_tf.inputs:execIn"),
            ("context.outputs:context", "pub_tf.inputs:context"),
            ("sim_time.outputs:simulationTime", "pub_tf.inputs:timeStamp"),
        ],
        keys.SET_VALUES: [
            ("sim_time.inputs:resetOnStop", True),
            ("pub_tf.inputs:topicName", "ground_truth_tf"),
            ("pub_tf.inputs:parentPrim", [usdrt.Sdf.Path("/World")]),
            ("pub_tf.inputs:targetPrims", [usdrt.Sdf.Path(ROBOT_PRIM)]),
        ],
    },
)
print(f"ground truth publisher ready: {GRAPH} -> /ground_truth_tf")
